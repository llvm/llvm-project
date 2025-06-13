// RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t.obj %s
// RUN: llvm-readobj --sections --sd --relocs %t.obj | FileCheck %s

.section        nc_sect,"xr"
normal_call:
.seh_proc normal_call
# %bb.0:                                # %entry
        subq    $40, %rsp
        .seh_stackalloc 40
        .seh_endprologue
.Limpcall0:
        rex64
        callq   *__imp_a(%rip)
        nopl    8(%rax,%rax)
        nop
        addq    $40, %rsp
        retq
        .seh_endproc

.section        tc_sect,"xr"
tail_call:
.Limpcall1:
        rex64
        jmp     *__imp_b(%rip)

.section        .retplne,"yi"
.asciz  "RetpolineV1"
.long   16
.secnum tc_sect
.long   2
.secoffset .Limpcall1
.long   16
.secnum nc_sect
.long   3
.secoffset .Limpcall0

// CHECK-LABEL: Name: .retplne (2E 72 65 74 70 6C 6E 65)
// CHECK-NEXT:  VirtualSize: 0x0
// CHECK-NEXT:  VirtualAddress: 0x0
// CHECK-NEXT:  RawDataSize: 44
// CHECK-NEXT:  PointerToRawData:
// CHECK-NEXT:  PointerToRelocations:
// CHECK-NEXT:  PointerToLineNumbers:
// CHECK-NEXT:  RelocationCount: 0
// CHECK-NEXT:  LineNumberCount: 0
// CHECK-NEXT:  Characteristics [
// CHECK-NEXT:    IMAGE_SCN_ALIGN_1BYTES
// CHECK-NEXT:    IMAGE_SCN_LNK_INFO
// CHECK-NEXT:  ]
// CHECK-NEXT:  SectionData (
// CHECK-NEXT:    52657470 6F6C696E 65563100 10000000  |RetpolineV1.....|
// CHECK-NEXT:    0010:
// CHECK-SAME:    [[#%.2X,TCSECT:]]000000
// CHECK-SAME:    02000000
// CHECK-SAME:    [[#%.2X,TCOFFSET:]]000000
// CHECK-SAME:    10000000
// CHECK-NEXT:    0020:
// CHECK-SAME:    [[#%.2X,NCSECT:]]000000
// CHECK-SAME:    03000000
// CHECK-SAME:    [[#%.2X,NCOFFSET:]]000000
// CHECK-NEXT:  )

// CHECK-LABEL: Relocations [
// CHECK-NEXT:     Section ([[#%u,NCSECT]]) nc_sect {
// CHECK-NEXT:       0x[[#%x,NCOFFSET + 3]] IMAGE_REL_AMD64_REL32 __imp_a
// CHECK-NEXT:     }
// CHECK-NEXT:     Section ([[#%u,TCSECT]]) tc_sect {
// CHECK-NEXT:       0x[[#%x,TCOFFSET + 3]] IMAGE_REL_AMD64_REL32 __imp_b
// CHECK-NEXT:     }
