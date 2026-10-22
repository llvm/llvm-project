/*
 * Packs random instructions with Mesa's own AGX packer (agx_pack.c) and
 * prints them, one per line, as "<generation> <name> <hex bytes>".
 *
 * The instructions are built the way Mesa's compiler builds them: operand
 * sizes, immediates and modes are limited to what agx_compile.c and its passes
 * create. mesa_isa.py --fuzz-check reads the output and checks that its
 * reference engine accepts every instruction, and that flipping any fixed bit
 * makes it reject the instruction.
 *
 * Build and run with tools/fuzz.sh MESA_SOURCE_DIR.
 */

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include "agx_pack.c"

/* Link stubs: the packer only needs these on its error and ralloc paths. */
void
agx_print_instr(const agx_instr *I, FILE *fp)
{
   fprintf(fp, "op %u\n", I->op);
}
void
ralloc_free(void *ptr)
{
   (void)ptr;
}
void *
reralloc_size(const void *ctx, void *ptr, size_t size)
{
   (void)ctx;
   return realloc(ptr, size);
}

static sigjmp_buf abort_jmp;
static void
on_abort(int sig)
{
   (void)sig;
   siglongjmp(abort_jmp, 1);
}

static unsigned
rnd(unsigned n)
{
   return n ? (unsigned)(random() % n) : 0;
}

static bool
coin(void)
{
   return rnd(2);
}

static enum agx_size
pick_size(unsigned sizes)
{
   for (;;) {
      unsigned s = rnd(3);
      if (sizes & (1u << s))
         return (enum agx_size)s;
   }
}

#define S16 1
#define S32 2
#define S64 4

static agx_index
reg(enum agx_size size)
{
   unsigned align = agx_size_align_16(size);
   unsigned v = rnd(256) & ~(align - 1);
   agx_index i = agx_register(v, size);
   return i;
}

static agx_index
hinted(agx_index i)
{
   unsigned h = rnd(3);
   i.cache = h == 1;
   i.discard = h == 2;
   return i;
}

enum src_kind { K_INT, K_IADD, K_FLOAT, K_HALF, K_CMPSEL };

static agx_index
alu_src(enum src_kind kind, unsigned sizes, enum agx_size dest_size)
{
   agx_index i;
   unsigned t = rnd(3);
   enum agx_size size;

   if (kind == K_CMPSEL)
      size = dest_size;
   else if (kind == K_HALF)
      size = AGX_SIZE_16;
   else
      size = pick_size(sizes);

   if (t == 0) {
      i = agx_immediate(rnd(256));
   } else if (t == 1 && size != AGX_SIZE_64) {
      i = agx_uniform(rnd(AGX_NUM_UNIFORMS), size);
   } else {
      i = hinted(reg(size));
   }

   if (kind == K_FLOAT || kind == K_HALF) {
      i.abs = coin();
      i.neg = coin();
   } else if (kind == K_IADD && i.type != AGX_INDEX_IMMEDIATE) {
      i.abs = coin(); /* zero-extend */
   }
   return i;
}

static agx_index
alu_dst(unsigned sizes)
{
   agx_index d = reg(pick_size(sizes));
   d.cache = coin();
   return d;
}

static const unsigned masks[] = {1, 3, 7, 15};
static const unsigned formats[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13};
static const unsigned iconds[] = {0, 1, 2, 4, 5, 6};
static const unsigned fconds[] = {0, 1, 2, 3, 5, 6, 7};
static const unsigned simd_ops[] = {0, 1, 2, 3, 4, 5, 7, 16, 20, 22, 28, 30};
static const unsigned srs[] = {0,  1,  2,  4,  5,  6,  8,  9,  10, 14, 20,
                               21, 48, 49, 50, 51, 52, 53, 56, 57, 58, 59,
                               60, 62, 63, 80, 81, 82, 124, 144, 146, 147};
static const unsigned atomic_ops[] = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const unsigned converts[] = {0, 1, 4, 5, 6, 7, 8, 9, 10, 11};
static const unsigned lod_modes[] = {0, 1, 2, 4, 5, 6, 9, 12, 13};
static const unsigned gathers[] = {0, 1, 3, 5, 7};

#define PICK(a) (a[rnd(sizeof(a) / sizeof(a[0]))])

static agx_index
mem_index(void)
{
   if (coin())
      return agx_immediate(rnd(0x10000));
   agx_index i = reg(AGX_SIZE_32);
   i.abs = coin();
   return i;
}

static agx_index
mem_base(void)
{
   unsigned v = rnd(256) & ~1;
   return coin() ? agx_uniform(v, AGX_SIZE_64) : agx_register(v, AGX_SIZE_64);
}

static agx_index
local_base(void)
{
   switch (rnd(3)) {
   case 0:
      return agx_zero();
   case 1:
      return agx_uniform(rnd(AGX_NUM_UNIFORMS), AGX_SIZE_16);
   default:
      return agx_register(rnd(256), AGX_SIZE_16);
   }
}

static agx_index
local_index(void)
{
   return coin() ? agx_immediate(rnd(0x10000))
                 : agx_register(rnd(256), AGX_SIZE_16);
}

static agx_index
mem_reg(void)
{
   return reg(pick_size(S16 | S32));
}

/* Fills I as Mesa's compiler would for op. Returns false for opcodes the
 * fuzzer does not build (pseudo-instructions). */
static bool
build(agx_instr *I, enum agx_opcode op, agx_index *src, agx_index *dest)
{
   struct agx_opcode_info info = agx_opcodes_info[op];
   memset(I, 0, sizeof(*I));
   I->op = op;
   I->src = src;
   I->dest = dest;
   I->nr_srcs = info.nr_srcs;
   I->nr_dests = info.nr_dests;
   for (unsigned i = 0; i < 8; ++i)
      src[i] = dest[i] = agx_null();

   const char *n = info.name;
   bool is_float = info.is_float;
   bool half = op == AGX_OPCODE_HADD || op == AGX_OPCODE_HMUL ||
               op == AGX_OPCODE_HFMA;
   bool i64 = op == AGX_OPCODE_IADD || op == AGX_OPCODE_IMAD;
   unsigned sizes = half ? S16 : i64 ? (S16 | S32 | S64) : (S16 | S32);

   if (info.encoding.exact == 0)
      return false;

   switch (op) {
   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_ST_TILE: {
      bool load = op == AGX_OPCODE_LD_TILE;
      I->format = PICK(formats);
      I->mask = 1 + rnd(15);
      I->pixel_offset = rnd(0x200);
      I->explicit_coords = coin();
      agx_index r = alu_dst(S16 | S32);
      agx_index sample =
         coin() ? agx_immediate(rnd(256)) : agx_register(rnd(256), AGX_SIZE_16);
      agx_index coords = I->explicit_coords ? reg(AGX_SIZE_32) : agx_null();
      if (load) {
         dest[0] = r;
         src[0] = sample;
         src[1] = coords;
      } else {
         src[0] = r;
         src[1] = sample;
         src[2] = coords;
      }
      return true;
   }
   case AGX_OPCODE_SAMPLE_MASK:
      src[0] = coin() ? agx_immediate(rnd(256)) : reg(AGX_SIZE_16);
      src[1] = coin() ? agx_immediate(rnd(256)) : reg(AGX_SIZE_16);
      return true;
   case AGX_OPCODE_WAIT:
      I->scoreboard = rnd(2);
      return true;
   case AGX_OPCODE_ITER:
   case AGX_OPCODE_ITERPROJ:
   case AGX_OPCODE_LDCF: {
      dest[0] = alu_dst(S16 | S32);
      I->channels = 1 + rnd(4);
      src[0] = coin() ? agx_immediate(rnd(256)) : agx_register(rnd(256), AGX_SIZE_16);
      unsigned next = 1;
      if (op == AGX_OPCODE_ITERPROJ)
         src[next++] = agx_immediate(rnd(256));
      if (op != AGX_OPCODE_LDCF) {
         I->interpolation = rnd(3);
         src[next] = I->interpolation == AGX_INTERPOLATION_SAMPLE && coin()
                        ? agx_register(rnd(256), AGX_SIZE_16)
                        : agx_immediate(rnd(256));
      }
      return true;
   }
   case AGX_OPCODE_ST_VARY:
      return false; /* not in agx_opcodes encoding table */
   case AGX_OPCODE_DEVICE_LOAD:
   case AGX_OPCODE_DEVICE_STORE: {
      bool store = op == AGX_OPCODE_DEVICE_STORE;
      I->format = PICK(formats);
      I->mask = PICK(masks);
      I->shift = rnd(3);
      I->coherent = coin();
      I->scoreboard = rnd(2);
      agx_index r = mem_reg();
      if (store) {
         src[0] = r;
         src[1] = mem_base();
         src[2] = mem_index();
      } else {
         dest[0] = r;
         src[0] = mem_base();
         src[1] = mem_index();
      }
      return true;
   }
   case AGX_OPCODE_UNIFORM_STORE: {
      unsigned sz = rnd(3);
      if (sz == 0) {
         src[0] = reg(AGX_SIZE_16);
         I->mask = PICK(masks);
      } else if (sz == 1) {
         src[0] = reg(AGX_SIZE_32);
         I->mask = coin() ? 1 : 3;
      } else {
         src[0] = reg(AGX_SIZE_64);
         I->mask = 1;
      }
      src[1] = mem_index();
      return true;
   }
   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_LOCAL_STORE: {
      bool load = op == AGX_OPCODE_LOCAL_LOAD;
      I->format = rnd(3);
      I->mask = PICK(masks);
      agx_index r = mem_reg();
      if (load) {
         dest[0] = r;
         src[0] = local_base();
         src[1] = local_index();
      } else {
         src[0] = r;
         src[1] = local_base();
         src[2] = local_index();
      }
      return true;
   }
   case AGX_OPCODE_ATOMIC:
      I->atomic_opc = PICK(atomic_ops);
      I->scoreboard = rnd(2);
      dest[0] = coin() ? reg(AGX_SIZE_32) : agx_null();
      src[0] = reg(AGX_SIZE_32);
      src[1] = mem_base();
      src[2] = mem_index();
      return true;
   case AGX_OPCODE_LOCAL_ATOMIC:
      I->atomic_opc = PICK(atomic_ops);
      dest[0] = coin() ? reg(AGX_SIZE_32) : agx_null();
      src[0] = reg(AGX_SIZE_32);
      src[1] = local_base();
      src[2] = agx_zero();
      return true;
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_TEXTURE_SAMPLE: {
      I->mask = 1 + rnd(15);
      I->dim = rnd(9);
      I->lod_mode = PICK(lod_modes);
      I->scoreboard = rnd(2);
      I->sparse = coin();
      I->offset = coin();
      I->gather = op == AGX_OPCODE_TEXTURE_SAMPLE ? PICK(gathers) : 0;
      I->shadow = op == AGX_OPCODE_TEXTURE_SAMPLE && coin();
      I->query_lod = op == AGX_OPCODE_TEXTURE_SAMPLE && coin();
      I->coherent = op == AGX_OPCODE_IMAGE_LOAD && coin();
      dest[0] = mem_reg();
      src[0] = reg(coin() ? AGX_SIZE_16 : AGX_SIZE_32);
      src[0].discard = coin();
      if (I->lod_mode == AGX_LOD_MODE_AUTO_LOD)
         src[1] = agx_zero();
      else if (!(I->lod_mode & 4)) {
         I->lod_mode |= 4;
         src[1] = agx_uniform(rnd(0x200), AGX_SIZE_16);
      } else
         src[1] = agx_register(rnd(256), AGX_SIZE_16);
      if (coin()) {
         src[2] = agx_zero();
         src[3] = coin() ? agx_immediate(rnd(256)) : agx_register(rnd(256), AGX_SIZE_16);
      } else {
         src[2] = agx_uniform(rnd(128) * 4, AGX_SIZE_64);
         src[3] = reg(AGX_SIZE_32);
      }
      src[4] = coin() ? agx_immediate(rnd(256)) : agx_register(rnd(256), AGX_SIZE_16);
      src[5] = coin() ? agx_null() : reg(AGX_SIZE_32);
      return true;
   }
   case AGX_OPCODE_TEX_STATE_STORE:
   case AGX_OPCODE_SAMPLER_STATE_STORE:
      src[0] = agx_uniform(rnd(16) * 4, AGX_SIZE_64);
      src[1] = coin() ? agx_immediate(rnd(128) * 2) : reg(AGX_SIZE_32);
      I->imm = rnd(128);
      I->scoreboard = rnd(2);
      return true;
   case AGX_OPCODE_IMAGE_WRITE:
      I->dim = rnd(9);
      I->coherent = coin();
      src[0] = reg(coin() ? AGX_SIZE_16 : AGX_SIZE_32);
      src[1] = reg(coin() ? AGX_SIZE_16 : AGX_SIZE_32);
      src[2] = coin() ? agx_immediate(rnd(64)) : agx_register(rnd(256), AGX_SIZE_16);
      if (coin()) {
         src[3] = agx_zero();
         src[4] = coin() ? agx_immediate(rnd(256)) : agx_register(rnd(256), AGX_SIZE_16);
      } else {
         src[3] = agx_uniform(rnd(8) * 4, AGX_SIZE_64);
         src[4] = reg(AGX_SIZE_32);
      }
      return true;
   case AGX_OPCODE_BLOCK_IMAGE_STORE:
      I->format = rnd(16);
      I->dim = rnd(9);
      I->explicit_coords = coin();
      if (coin()) {
         src[0] = agx_zero();
         src[1] = coin() ? agx_immediate(rnd(256)) : agx_register(rnd(256), AGX_SIZE_16);
      } else {
         src[0] = agx_uniform(rnd(8) * 4, AGX_SIZE_64);
         src[1] = reg(AGX_SIZE_32);
      }
      src[2] = reg(AGX_SIZE_32);
      src[3] = reg(AGX_SIZE_16);
      src[3].discard = coin();
      return true;
   case AGX_OPCODE_ZS_EMIT:
      I->zs = 1 + rnd(3);
      src[0] = coin() ? agx_immediate(rnd(256)) : reg(AGX_SIZE_16);
      src[1] = reg(AGX_SIZE_32);
      return true;
   case AGX_OPCODE_JMP_EXEC_ANY:
   case AGX_OPCODE_JMP_EXEC_NONE:
   case AGX_OPCODE_JMP_EXEC_NONE_AFTER:
      I->target = (agx_block *)(uintptr_t)1;
      return true;
   case AGX_OPCODE_DOORBELL:
      I->imm = rnd(255);
      return true;
   case AGX_OPCODE_STACK_UNMAP:
      dest[0] = reg(AGX_SIZE_32);
      I->imm = rnd(0xffff);
      return true;
   case AGX_OPCODE_STACK_MAP:
      src[0] = reg(AGX_SIZE_32);
      I->imm = rnd(0xffff);
      return true;
   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_STACK_STORE: {
      bool load = op == AGX_OPCODE_STACK_LOAD;
      I->format = rnd(3);
      I->mask = PICK(masks);
      I->scoreboard = rnd(2);
      agx_index r = mem_reg();
      agx_index off = mem_index();
      off.abs = false;
      if (load) {
         dest[0] = r;
         src[0] = off;
      } else {
         src[0] = r;
         src[1] = off;
      }
      return true;
   }
   case AGX_OPCODE_STACK_ADJUST:
      I->stack_size = (int16_t)(rnd(0x10000) - 0x8000);
      return true;
   default:
      break;
   }

   /* ALU */
   if (op == AGX_OPCODE_EXPORT)
      return false;

   bool cmpsel = op == AGX_OPCODE_ICMPSEL || op == AGX_OPCODE_FCMPSEL;
   enum agx_size dsize = AGX_SIZE_32;
   if (info.nr_dests) {
      if (op == AGX_OPCODE_MOV_IMM) {
         dest[0] = alu_dst(S16 | S32);
         I->imm = dest[0].size == AGX_SIZE_16 ? rnd(0x10000) : (uint32_t)random();
      } else {
         dest[0] = alu_dst(sizes);
      }
      dsize = dest[0].size;
   }
   if (cmpsel && dsize == AGX_SIZE_64)
      return false;

   for (unsigned s = 0; s < info.nr_srcs; ++s) {
      enum src_kind kind = K_INT;
      if (cmpsel && s >= 2)
         kind = K_CMPSEL;
      else if (half)
         kind = K_HALF;
      else if (is_float || op == AGX_OPCODE_FCMPSEL)
         kind = K_FLOAT;
      else if (i64)
         kind = K_IADD;
      src[s] = alu_src(kind, sizes, dsize);
   }

   if (op == AGX_OPCODE_CONVERT) {
      src[0] = agx_immediate(PICK(converts));
      I->round = rnd(2);
   }
   if (op == AGX_OPCODE_IADD && coin())
      src[1].neg = true;
   if (op == AGX_OPCODE_IMAD && coin())
      src[2].neg = true;

   bool saturable = (is_float && info.nr_dests) || op == AGX_OPCODE_IADD;
   I->saturate = saturable && coin();

   if (info.immediates & AGX_IMMEDIATE_SHIFT)
      I->shift = rnd(5);
   if (info.immediates & AGX_IMMEDIATE_TRUTH_TABLE)
      I->truth_table = rnd(16);
   if (info.immediates & AGX_IMMEDIATE_BFI_MASK)
      I->bfi_mask = rnd(32);
   if (info.immediates & AGX_IMMEDIATE_SIMD_OP)
      I->simd_op = PICK(simd_ops);
   if (info.immediates & AGX_IMMEDIATE_SR)
      I->sr = PICK(srs);
   if (info.immediates & AGX_IMMEDIATE_WRITEOUT)
      I->imm = rnd(256);
   if (info.immediates & AGX_IMMEDIATE_ICOND)
      I->icond = PICK(iconds);
   if (info.immediates & AGX_IMMEDIATE_FCOND)
      I->fcond = PICK(fconds);
   if (info.immediates & AGX_IMMEDIATE_INVERT_COND)
      I->invert_cond = coin();
   if (info.immediates & AGX_IMMEDIATE_NEST)
      I->nest = rnd(4);
   (void)n;
   return true;
}

int
main(int argc, char **argv)
{
   unsigned count = argc > 1 ? atoi(argv[1]) : 200;
   srandom(argc > 2 ? atoi(argv[2]) : 1);
   signal(SIGABRT, on_abort);

   static const struct {
      const char *name;
      enum u_tristate coherency;
   } gens[] = {{"g13", U_TRISTATE_NO}, {"g13x", U_TRISTATE_YES},
               {"g14", U_TRISTATE_NO}, {"g14d", U_TRISTATE_YES}};

   agx_index src[8], dest[8];
   agx_instr I;
   for (unsigned op = 0; op < AGX_NUM_OPCODES; ++op) {
      unsigned made = 0, tries = 0;
      while (made < count && tries < count * 20) {
         tries++;
         if (!build(&I, op, src, dest))
            break;
         unsigned g = rnd(4);
         struct util_dynarray em, fix;
         util_dynarray_init(&em, NULL);
         util_dynarray_init(&fix, NULL);
         if (sigsetjmp(abort_jmp, 1)) {
            signal(SIGABRT, on_abort);
            continue; /* the packer refused this combination */
         }
         agx_pack_instr(&em, &fix, &I, gens[g].coherency);
         printf("%s %s ", gens[g].name, agx_opcodes_info[op].name);
         for (unsigned i = 0; i < em.size; ++i)
            printf("%02x", ((uint8_t *)em.data)[i]);
         printf("\n");
         made++;
         free(em.data);
         free(fix.data);
      }
   }
   return 0;
}
