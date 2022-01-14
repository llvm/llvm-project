# clang-format off
# REQUIRES: lld, x86

# RUN: llvm-mc -triple=x86_64-windows-msvc --filetype=obj %s > %t.obj
# RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj  -out:%t.exe
# RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
# RUN:     %p/Inputs/inline_sites.lldbinit 2>&1 | FileCheck %s

# Compiled from the following files, but replaced the call to abort with nop.
# a.cpp:
# #include "stdlib.h"
# #include "a.h"
# int main(int argc, char** argv) {
#   Namespace1::foo(2);
#   return 0;
# }
# a.h:
# #include "b.h"
# namespace Namespace1 {
# inline void foo(int x) {
#   static volatile int gv_foo;
#   ++gv_foo;
#   if (!gv_foo)
#     abort();
#   Class1::bar(x + 1);
# }
# }
# b.h:
# #include "c.h"
# class Class1 {
# public:
#   inline static void bar(int x) {
#     static volatile int gv_bar;
#     ++gv_bar;
#     Namespace2::Class2::func(x + 1);
#   }
# };
# c.h:
# namespace Namespace2{
#   class Class2{
#     public:
#     inline static void func(int x) {
#       static volatile int gv_func;
#       gv_func += x;
#     }
#   };
# }

# CHECK:      (lldb) image dump line-table a.cpp -v
# CHECK-NEXT: Line table for {{.*}}a.cpp in
# CHECK-NEXT: 0x0000000140001000: {{.*}}a.cpp:3
# CHECK-NEXT: 0x0000000140001004: {{.*}}a.h:5, is_start_of_statement = TRUE, is_prologue_end = TRUE
# CHECK-NEXT: 0x000000014000100a: {{.*}}a.h:6
# CHECK-NEXT: 0x0000000140001014: {{.*}}b.h:6, is_start_of_statement = TRUE, is_prologue_end = TRUE
# CHECK-NEXT: 0x000000014000101a: {{.*}}c.h:6, is_start_of_statement = TRUE, is_prologue_end = TRUE
# CHECK-NEXT: 0x0000000140001021: {{.*}}a.cpp:5
# CHECK-NEXT: 0x0000000140001028: {{.*}}a.h:7, is_start_of_statement = TRUE
# CHECK-NEXT: 0x000000014000102a: {{.*}}a.cpp:5, is_terminal_entry = TRUE

# CEHCK: (lldb) b a.cpp:5
# CHECK: Breakpoint 1: where = {{.*}}`main + 33 at a.cpp:5, address = 0x0000000140001021
# CEHCK: (lldb) b a.h:5
# CHECK: Breakpoint 2: where = {{.*}}`main + 4 [inlined] Namespace1::foo at a.h:5, address = 0x0000000140001004
# CEHCK: (lldb) b a.h:6
# CHECK: Breakpoint 3: where = {{.*}}`main + 10 [inlined] Namespace1::foo + 6 at a.h:6, address = 0x000000014000100a
# CEHCK: (lldb) b a.h:7
# CHECK: Breakpoint 4: where = {{.*}}`main + 40 [inlined] Namespace1::foo at a.h:7, address = 0x0000000140001028
# CEHCK: (lldb) b b.h:6
# CHECK: Breakpoint 5: where = {{.*}}`main + 20 [inlined] Class1::bar at b.h:6, address = 0x0000000140001014
# CEHCK: (lldb) b c.h:6
# CHECK: Breakpoint 6: where = {{.*}}`main + 26 [inlined] Namespace2::Class2::func at c.h:6, address = 0x000000014000101a

# CEHCK-LABEL: (lldb) image lookup -a 0x140001003 -v
# CHECK:      Summary: {{.*}}`main + 3 at a.cpp:3
# CHECK:     Function: id = {{.*}}, name = "main", range = [0x0000000140001000-0x000000014000102a)
# CHECK:       Blocks: id = {{.*}}, range = [0x140001000-0x14000102a)
# CHECK:    LineEntry: [0x0000000140001000-0x0000000140001004): {{.*}}a.cpp:3

# CEHCK-LABEL: (lldb) image lookup -a 0x140001004 -v
# CHECK:      Summary: {{.*}}`main + 4 [inlined] Namespace1::foo at a.h:5
# CHECK-NEXT:          {{.*}}`main + 4 at a.cpp:4
# CHECK:     Function: id = {{.*}}, name = "main", range = [0x0000000140001000-0x000000014000102a)
# CHECK:       Blocks: id = {{.*}}, range = [0x140001000-0x14000102a)
# CHECK-NEXT:          id = {{.*}}, ranges = [0x140001004-0x140001021)[0x140001028-0x14000102a), name = "Namespace1::foo", decl = a.h:3
# CHECK:    LineEntry: [0x0000000140001004-0x000000014000100a): {{.*}}a.h:5

# CEHCK-LABEL: (lldb) image lookup -a 0x140001014 -v
# CHECK:      Summary: {{.*}}`main + 20 [inlined] Class1::bar at b.h:6
# CHECK-NEXT:          {{.*}}`main + 20 [inlined] Namespace1::foo + 16 at a.h:8
# CHECK-NEXT:          {{.*}}`main + 4 at a.cpp:4
# CHECK:     Function: id = {{.*}}, name = "main", range = [0x0000000140001000-0x000000014000102a)
# CHECK:       Blocks: id = {{.*}}, range = [0x140001000-0x14000102a)
# CHECK-NEXT:          id = {{.*}}, ranges = [0x140001004-0x140001021)[0x140001028-0x14000102a), name = "Namespace1::foo", decl = a.h:3
# CHECK-NEXT:          id = {{.*}}, range = [0x140001014-0x140001021), name = "Class1::bar", decl = b.h:4
# CHECK:    LineEntry: [0x0000000140001014-0x000000014000101a): {{.*}}b.h:6

# CEHCK-LABEL: (lldb) image lookup -a 0x14000101a -v
# CHECK:      Summary: {{.*}}`main + 26 [inlined] Namespace2::Class2::func at c.h:6
# CHECK-NEXT:          {{.*}}`main + 26 [inlined] Class1::bar + 6 at b.h:7
# CHECK-NEXT:          {{.*}}`main + 20 [inlined] Namespace1::foo + 16 at a.h:8
# CHECK-NEXT:          {{.*}}`main + 4 at a.cpp:4
# CHECK:     Function: id = {{.*}}, name = "main", range = [0x0000000140001000-0x000000014000102a)
# CHECK:       Blocks: id = {{.*}}, range = [0x140001000-0x14000102a)
# CHECK-NEXT:          id = {{.*}}, ranges = [0x140001004-0x140001021)[0x140001028-0x14000102a), name = "Namespace1::foo", decl = a.h:3
# CHECK-NEXT:          id = {{.*}}, range = [0x140001014-0x140001021), name = "Class1::bar", decl = b.h:4
# CHECK-NEXT:          id = {{.*}}, range = [0x14000101a-0x140001021), name = "Namespace2::Class2::func", decl = c.h:4
# CHECK:    LineEntry: [0x000000014000101a-0x0000000140001021): {{.*}}c.h:6

# CEHCK-LABEL: (lldb) image lookup -a 0x140001021 -v
# CHECK:      Summary: {{.*}}`main + 33 at a.cpp:5
# CHECK:     Function: id = {{.*}}, name = "main", range = [0x0000000140001000-0x000000014000102a)
# CHECK:       Blocks: id = {{.*}}, range = [0x140001000-0x14000102a)
# CHECK:    LineEntry: [0x0000000140001021-0x0000000140001028): {{.*}}a.cpp:5

# CEHCK-LABEL: (lldb) image lookup -a 0x140001028 -v
# CHECK:      Summary: {{.*}}`main + 40 [inlined] Namespace1::foo at a.h:7
# CHECK-NEXT:          {{.*}}`main + 40 at a.cpp:4
# CHECK:     Function: id = {{.*}}, name = "main", range = [0x0000000140001000-0x000000014000102a)
# CHECK:       Blocks: id = {{.*}}, range = [0x140001000-0x14000102a)
# CHECK-NEXT:          id = {{.*}}, ranges = [0x140001004-0x140001021)[0x140001028-0x14000102a), name = "Namespace1::foo", decl = a.h:3
# CHECK:    LineEntry: [0x0000000140001028-0x000000014000102a): {{.*}}a.h:7

	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.intel_syntax noprefix
	.file	"a.cpp"
	.def	 main;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,main
	.globl	main                            # -- Begin function main
main:                                   # @main
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "/tmp/a.cpp" "4ECCDD2814054DCF80EA72F4349036C4" 1
	.cv_loc	0 1 3 0                         # a.cpp:3:0
.seh_proc main
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argv <- $rdx
	#DEBUG_VALUE: main:argc <- $ecx
	#DEBUG_VALUE: foo:x <- 2
	sub	rsp, 40
	.seh_stackalloc 40
	.seh_endprologue
.Ltmp0:
	.cv_file	2 "/tmp/./a.h" "9E656AFA1B1B681265C87EEA8BBE073E" 1
	.cv_inline_site_id 1 within 0 inlined_at 1 4 0
	.cv_loc	1 2 5 0                         # ./a.h:5:0
	inc	dword ptr [rip + "?gv_foo@?1??foo@Namespace1@@YAXH@Z@4HC"]
	.cv_loc	1 2 6 0                         # ./a.h:6:0
	mov	eax, dword ptr [rip + "?gv_foo@?1??foo@Namespace1@@YAXH@Z@4HC"]
	test	eax, eax
	je	.LBB0_2
.Ltmp1:
# %bb.1:                                # %"?foo@Namespace1@@YAXH@Z.exit"
	#DEBUG_VALUE: foo:x <- 2
	#DEBUG_VALUE: main:argc <- $ecx
	#DEBUG_VALUE: main:argv <- $rdx
	#DEBUG_VALUE: bar:x <- [DW_OP_plus_uconst 1, DW_OP_stack_value] 2
	.cv_file	3 "/tmp/./b.h" "BE52983EB17A3B0DA14E68A5CCBC4399" 1
	.cv_inline_site_id 2 within 1 inlined_at 2 8 0
	.cv_loc	2 3 6 0                         # ./b.h:6:0
	inc	dword ptr [rip + "?gv_bar@?1??bar@Class1@@SAXH@Z@4HC"]
.Ltmp2:
	#DEBUG_VALUE: func:x <- 4
	.cv_file	4 "/tmp/./c.h" "D1B76A1C2A54DBEA648F3A11496166B8" 1
	.cv_inline_site_id 3 within 2 inlined_at 3 7 0
	.cv_loc	3 4 6 0                         # ./c.h:6:0
	add	dword ptr [rip + "?gv_func@?1??func@Class2@Namespace2@@SAXH@Z@4HC"], 4
.Ltmp3:
	.cv_loc	0 1 5 0                         # a.cpp:5:0
	xor	eax, eax
	add	rsp, 40
	ret
.Ltmp4:
.LBB0_2:                                # %if.then.i
	#DEBUG_VALUE: foo:x <- 2
	#DEBUG_VALUE: main:argc <- $ecx
	#DEBUG_VALUE: main:argv <- $rdx
	.cv_loc	1 2 7 0                         # ./a.h:7:0
	nop
.Ltmp5:
	int3
.Ltmp6:
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rdx
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $ecx
.Lfunc_end0:
	.seh_endproc
                                        # -- End function
	.section	.bss,"bw",discard,"?gv_foo@?1??foo@Namespace1@@YAXH@Z@4HC"
	.globl	"?gv_foo@?1??foo@Namespace1@@YAXH@Z@4HC" # @"?gv_foo@?1??foo@Namespace1@@YAXH@Z@4HC"
	.p2align	2
"?gv_foo@?1??foo@Namespace1@@YAXH@Z@4HC":
	.long	0                               # 0x0

	.section	.bss,"bw",discard,"?gv_bar@?1??bar@Class1@@SAXH@Z@4HC"
	.globl	"?gv_bar@?1??bar@Class1@@SAXH@Z@4HC" # @"?gv_bar@?1??bar@Class1@@SAXH@Z@4HC"
	.p2align	2
"?gv_bar@?1??bar@Class1@@SAXH@Z@4HC":
	.long	0                               # 0x0

	.section	.bss,"bw",discard,"?gv_func@?1??func@Class2@Namespace2@@SAXH@Z@4HC"
	.globl	"?gv_func@?1??func@Class2@Namespace2@@SAXH@Z@4HC" # @"?gv_func@?1??func@Class2@Namespace2@@SAXH@Z@4HC"
	.p2align	2
"?gv_func@?1??func@Class2@Namespace2@@SAXH@Z@4HC":
	.long	0                               # 0x0

	.section	.drectve,"yn"
	.ascii	" /DEFAULTLIB:libcmt.lib"
	.ascii	" /DEFAULTLIB:oldnames.lib"
	.section	.debug$S,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp8-.Ltmp7                   # Subsection size
.Ltmp7:
	.short	.Ltmp10-.Ltmp9                  # Record length
.Ltmp9:
	.short	4353                            # Record kind: S_OBJNAME
	.long	0                               # Signature
	.asciz	"/tmp/a-e5dd01.obj"             # Object name
	.p2align	2
.Ltmp10:
	.short	.Ltmp12-.Ltmp11                 # Record length
.Ltmp11:
	.short	4412                            # Record kind: S_COMPILE3
	.long	1                               # Flags and language
	.short	208                             # CPUType
	.short	14                              # Frontend version
	.short	0
	.short	0
	.short	0
	.short	14000                           # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 14.0.0"          # Null-terminated compiler version string
	.p2align	2
.Ltmp12:
.Ltmp8:
	.p2align	2
	.long	246                             # Inlinee lines subsection
	.long	.Ltmp14-.Ltmp13                 # Subsection size
.Ltmp13:
	.long	0                               # Inlinee lines signature

                                        # Inlined function foo starts at ./a.h:3
	.long	4099                            # Type index of inlined function
	.cv_filechecksumoffset	2               # Offset into filechecksum table
	.long	3                               # Starting line number

                                        # Inlined function bar starts at ./b.h:4
	.long	4106                            # Type index of inlined function
	.cv_filechecksumoffset	3               # Offset into filechecksum table
	.long	4                               # Starting line number

                                        # Inlined function func starts at ./c.h:4
	.long	4113                            # Type index of inlined function
	.cv_filechecksumoffset	4               # Offset into filechecksum table
	.long	4                               # Starting line number
.Ltmp14:
	.p2align	2
	.section	.debug$S,"dr",associative,main
	.p2align	2
	.long	4                               # Debug section magic
	.long	241                             # Symbol subsection for main
	.long	.Ltmp16-.Ltmp15                 # Subsection size
.Ltmp15:
	.short	.Ltmp18-.Ltmp17                 # Record length
.Ltmp17:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end0-main                # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4117                            # Function type index
	.secrel32	main                    # Function section relative address
	.secidx	main                            # Function section index
	.byte	0                               # Flags
	.asciz	"main"                          # Function name
	.p2align	2
.Ltmp18:
	.short	.Ltmp20-.Ltmp19                 # Record length
.Ltmp19:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	40                              # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	81920                           # Flags (defines frame register)
	.p2align	2
.Ltmp20:
	.short	.Ltmp22-.Ltmp21                 # Record length
.Ltmp21:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	1                               # Flags
	.asciz	"argc"
	.p2align	2
.Ltmp22:
	.cv_def_range	 .Lfunc_begin0 .Ltmp5, reg, 18
	.short	.Ltmp24-.Ltmp23                 # Record length
.Ltmp23:
	.short	4414                            # Record kind: S_LOCAL
	.long	4114                            # TypeIndex
	.short	1                               # Flags
	.asciz	"argv"
	.p2align	2
.Ltmp24:
	.cv_def_range	 .Lfunc_begin0 .Ltmp5, reg, 331
	.short	.Ltmp26-.Ltmp25                 # Record length
.Ltmp25:
	.short	4365                            # Record kind: S_GDATA32
	.long	4118                            # Type
	.secrel32	"?gv_foo@?1??foo@Namespace1@@YAXH@Z@4HC" # DataOffset
	.secidx	"?gv_foo@?1??foo@Namespace1@@YAXH@Z@4HC" # Segment
	.asciz	"Namespace1::foo::gv_foo"       # Name
	.p2align	2
.Ltmp26:
	.short	.Ltmp28-.Ltmp27                 # Record length
.Ltmp27:
	.short	4365                            # Record kind: S_GDATA32
	.long	4118                            # Type
	.secrel32	"?gv_bar@?1??bar@Class1@@SAXH@Z@4HC" # DataOffset
	.secidx	"?gv_bar@?1??bar@Class1@@SAXH@Z@4HC" # Segment
	.asciz	"Class1::bar::gv_bar"           # Name
	.p2align	2
.Ltmp28:
	.short	.Ltmp30-.Ltmp29                 # Record length
.Ltmp29:
	.short	4365                            # Record kind: S_GDATA32
	.long	4118                            # Type
	.secrel32	"?gv_func@?1??func@Class2@Namespace2@@SAXH@Z@4HC" # DataOffset
	.secidx	"?gv_func@?1??func@Class2@Namespace2@@SAXH@Z@4HC" # Segment
	.asciz	"Namespace2::Class2::func::gv_func" # Name
	.p2align	2
.Ltmp30:
	.short	.Ltmp32-.Ltmp31                 # Record length
.Ltmp31:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4099                            # Inlinee type index
	.cv_inline_linetable	1 2 3 .Lfunc_begin0 .Lfunc_end0
	.p2align	2
.Ltmp32:
	.short	.Ltmp34-.Ltmp33                 # Record length
.Ltmp33:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	257                             # Flags
	.asciz	"x"
	.p2align	2
.Ltmp34:
	.short	.Ltmp36-.Ltmp35                 # Record length
.Ltmp35:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4106                            # Inlinee type index
	.cv_inline_linetable	2 3 4 .Lfunc_begin0 .Lfunc_end0
	.p2align	2
.Ltmp36:
	.short	.Ltmp38-.Ltmp37                 # Record length
.Ltmp37:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	257                             # Flags
	.asciz	"x"
	.p2align	2
.Ltmp38:
	.short	.Ltmp40-.Ltmp39                 # Record length
.Ltmp39:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4113                            # Inlinee type index
	.cv_inline_linetable	3 4 4 .Lfunc_begin0 .Lfunc_end0
	.p2align	2
.Ltmp40:
	.short	.Ltmp42-.Ltmp41                 # Record length
.Ltmp41:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	257                             # Flags
	.asciz	"x"
	.p2align	2
.Ltmp42:
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp16:
	.p2align	2
	.cv_linetable	0, main, .Lfunc_end0
	.section	.debug$S,"dr"
	.long	241
	.long	.Ltmp44-.Ltmp43                 # Subsection size
.Ltmp43:
	.short	.Ltmp46-.Ltmp45                 # Record length
.Ltmp45:
	.short	4360                            # Record kind: S_UDT
	.long	4103                            # Type
	.asciz	"Class1"
	.p2align	2
.Ltmp46:
	.short	.Ltmp48-.Ltmp47                 # Record length
.Ltmp47:
	.short	4360                            # Record kind: S_UDT
	.long	4110                            # Type
	.asciz	"Namespace2::Class2"
	.p2align	2
.Ltmp48:
.Ltmp44:
	.p2align	2
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
	.long	241
	.long	.Ltmp50-.Ltmp49                 # Subsection size
.Ltmp49:
	.short	.Ltmp52-.Ltmp51                 # Record length
.Ltmp51:
	.short	4428                            # Record kind: S_BUILDINFO
	.long	4121                            # LF_BUILDINFO index
	.p2align	2
.Ltmp52:
.Ltmp50:
	.p2align	2
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	# StringId (0x1000)
	.short	0x12                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"Namespace1"                    # StringData
	.byte	241
	# ArgList (0x1001)
	.short	0xa                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x1                             # NumArgs
	.long	0x74                            # Argument: int
	# Procedure (0x1002)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x3                             # ReturnType: void
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x1                             # NumParameters
	.long	0x1001                          # ArgListType: (int)
	# FuncId (0x1003)
	.short	0xe                             # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x1000                          # ParentScope: Namespace1
	.long	0x1002                          # FunctionType: void (int)
	.asciz	"foo"                           # Name
	# Class (0x1004)
	.short	0x2a                            # Record length
	.short	0x1504                          # Record kind: LF_CLASS
	.short	0x0                             # MemberCount
	.short	0x280                           # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) )
	.long	0x0                             # FieldList
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x0                             # SizeOf
	.asciz	"Class1"                        # Name
	.asciz	".?AVClass1@@"                  # LinkageName
	.byte	242
	.byte	241
	# MemberFunction (0x1005)
	.short	0x1a                            # Record length
	.short	0x1009                          # Record kind: LF_MFUNCTION
	.long	0x3                             # ReturnType: void
	.long	0x1004                          # ClassType: Class1
	.long	0x0                             # ThisType
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x1                             # NumParameters
	.long	0x1001                          # ArgListType: (int)
	.long	0x0                             # ThisAdjustment
	# FieldList (0x1006)
	.short	0xe                             # Record length
	.short	0x1203                          # Record kind: LF_FIELDLIST
	.short	0x1511                          # Member kind: OneMethod ( LF_ONEMETHOD )
	.short	0xb                             # Attrs: Public, Static
	.long	0x1005                          # Type: void Class1::(int)
	.asciz	"bar"                           # Name
	# Class (0x1007)
	.short	0x2a                            # Record length
	.short	0x1504                          # Record kind: LF_CLASS
	.short	0x1                             # MemberCount
	.short	0x200                           # Properties ( HasUniqueName (0x200) )
	.long	0x1006                          # FieldList: <field list>
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x1                             # SizeOf
	.asciz	"Class1"                        # Name
	.asciz	".?AVClass1@@"                  # LinkageName
	.byte	242
	.byte	241
	# StringId (0x1008)
	.short	0x12                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"/tmp/./b.h"                    # StringData
	.byte	241
	# UdtSourceLine (0x1009)
	.short	0xe                             # Record length
	.short	0x1606                          # Record kind: LF_UDT_SRC_LINE
	.long	0x1007                          # UDT: Class1
	.long	0x1008                          # SourceFile: /tmp/./b.h
	.long	0x2                             # LineNumber
	# MemberFuncId (0x100A)
	.short	0xe                             # Record length
	.short	0x1602                          # Record kind: LF_MFUNC_ID
	.long	0x1004                          # ClassType: Class1
	.long	0x1005                          # FunctionType: void Class1::(int)
	.asciz	"bar"                           # Name
	# Class (0x100B)
	.short	0x42                            # Record length
	.short	0x1504                          # Record kind: LF_CLASS
	.short	0x0                             # MemberCount
	.short	0x280                           # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) )
	.long	0x0                             # FieldList
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x0                             # SizeOf
	.asciz	"Namespace2::Class2"            # Name
	.asciz	".?AVClass2@Namespace2@@"       # LinkageName
	.byte	243
	.byte	242
	.byte	241
	# MemberFunction (0x100C)
	.short	0x1a                            # Record length
	.short	0x1009                          # Record kind: LF_MFUNCTION
	.long	0x3                             # ReturnType: void
	.long	0x100b                          # ClassType: Namespace2::Class2
	.long	0x0                             # ThisType
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x1                             # NumParameters
	.long	0x1001                          # ArgListType: (int)
	.long	0x0                             # ThisAdjustment
	# FieldList (0x100D)
	.short	0x12                            # Record length
	.short	0x1203                          # Record kind: LF_FIELDLIST
	.short	0x1511                          # Member kind: OneMethod ( LF_ONEMETHOD )
	.short	0xb                             # Attrs: Public, Static
	.long	0x100c                          # Type: void Namespace2::Class2::(int)
	.asciz	"func"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# Class (0x100E)
	.short	0x42                            # Record length
	.short	0x1504                          # Record kind: LF_CLASS
	.short	0x1                             # MemberCount
	.short	0x200                           # Properties ( HasUniqueName (0x200) )
	.long	0x100d                          # FieldList: <field list>
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x1                             # SizeOf
	.asciz	"Namespace2::Class2"            # Name
	.asciz	".?AVClass2@Namespace2@@"       # LinkageName
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x100F)
	.short	0x12                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"/tmp/./c.h"                    # StringData
	.byte	241
	# UdtSourceLine (0x1010)
	.short	0xe                             # Record length
	.short	0x1606                          # Record kind: LF_UDT_SRC_LINE
	.long	0x100e                          # UDT: Namespace2::Class2
	.long	0x100f                          # SourceFile: /tmp/./c.h
	.long	0x2                             # LineNumber
	# MemberFuncId (0x1011)
	.short	0x12                            # Record length
	.short	0x1602                          # Record kind: LF_MFUNC_ID
	.long	0x100b                          # ClassType: Namespace2::Class2
	.long	0x100c                          # FunctionType: void Namespace2::Class2::(int)
	.asciz	"func"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# Pointer (0x1012)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x670                           # PointeeType: char*
	.long	0x1000c                         # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
	# ArgList (0x1013)
	.short	0xe                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x2                             # NumArgs
	.long	0x74                            # Argument: int
	.long	0x1012                          # Argument: char**
	# Procedure (0x1014)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x2                             # NumParameters
	.long	0x1013                          # ArgListType: (int, char**)
	# FuncId (0x1015)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1014                          # FunctionType: int (int, char**)
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# Modifier (0x1016)
	.short	0xa                             # Record length
	.short	0x1001                          # Record kind: LF_MODIFIER
	.long	0x74                            # ModifiedType: int
	.short	0x2                             # Modifiers ( Volatile (0x2) )
	.byte	242
	.byte	241
	# StringId (0x1017)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"/tmp"                          # StringData
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x1018)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"a.cpp"                         # StringData
	.byte	242
	.byte	241
	# BuildInfo (0x1019)
	.short	0x1a                            # Record length
	.short	0x1603                          # Record kind: LF_BUILDINFO
	.short	0x5                             # NumArgs
	.long	0x1017                          # Argument: /tmp
	.long	0x0                             # Argument
	.long	0x1018                          # Argument: a.cpp
	.long	0x0                             # Argument
	.long	0x0                             # Argument
	.byte	242
	.byte	241
	.addrsig
	.addrsig_sym "?gv_foo@?1??foo@Namespace1@@YAXH@Z@4HC"
	.addrsig_sym "?gv_bar@?1??bar@Class1@@SAXH@Z@4HC"
	.addrsig_sym "?gv_func@?1??func@Class2@Namespace2@@SAXH@Z@4HC"
