// RUN: %llvmgcc %s -S -g

typedef long unsigned int size_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
typedef uint16_t Elf64_Half;
typedef uint32_t Elf64_Word;
typedef uint64_t Elf64_Xword;
typedef uint64_t Elf64_Addr;
typedef uint64_t Elf64_Off;
typedef struct
{
  Elf64_Word p_type;
  Elf64_Off p_offset;
  Elf64_Addr p_vaddr;
  Elf64_Xword p_align;
}
Elf64_Phdr;
struct dl_phdr_info
{
  const char *dlpi_name;
  const Elf64_Phdr *dlpi_phdr;
  Elf64_Half dlpi_phnum;
  unsigned long long int dlpi_adds;
};
typedef unsigned _Unwind_Ptr;
struct object
{
  union
  {
    const struct dwarf_fde *single;
    struct dwarf_fde **array;
    struct fde_vector *sort;
  }
  u;
  union
  {
    struct
    {
    }
    b;
  }
  s;
  struct object *next;
};
typedef int sword;
typedef unsigned int uword;
struct dwarf_fde
{
  uword length;
  sword CIE_delta;
  unsigned char pc_begin[];
};
typedef struct dwarf_fde fde;
struct unw_eh_callback_data
{
  const fde *ret;
  struct frame_hdr_cache_element *link;
}
frame_hdr_cache[8];

_Unwind_Ptr
base_from_cb_data (struct unw_eh_callback_data *data)
{
}

void
_Unwind_IteratePhdrCallback (struct dl_phdr_info *info, size_t size, void *ptr)
{
  const unsigned char *p;
  const struct unw_eh_frame_hdr *hdr;
  struct object ob;
}
