// RUN: llvm-mc -filetype=obj -triple x86_64-mingw32 %s -o - | llvm-objdump -r - | FileCheck --check-prefix=COFF_X86_64 %s
// RUN: llvm-mc -filetype=obj -triple i686-mingw32 %s -o - | llvm-objdump -r - | FileCheck --check-prefix=COFF_I686 %s

.cfi_sections .debug_frame

f1:
        .cfi_startproc
        nop
        .cfi_endproc

f2:
        .cfi_startproc
        nop
        .cfi_endproc

// COFF_X86_64: RELOCATION RECORDS FOR [.debug_frame]:
// COFF_X86_64-NEXT: {{.*}}OFFSET TYPE VALUE
// COFF_X86_64-NEXT: {{.*}} IMAGE_REL_AMD64_SECREL .debug_frame
// COFF_X86_64-NEXT: {{.*}} IMAGE_REL_AMD64_ADDR64 .text
// COFF_X86_64-NEXT: {{.*}} IMAGE_REL_AMD64_SECREL .debug_frame
// COFF_X86_64-NEXT: {{.*}} IMAGE_REL_AMD64_ADDR64 .text

// COFF_I686: RELOCATION RECORDS FOR [.debug_frame]:
// COFF_I686-NEXT: {{.*}}OFFSET TYPE VALUE
// COFF_I686-NEXT: {{.*}} IMAGE_REL_I386_SECREL .debug_frame
// COFF_I686-NEXT: {{.*}} IMAGE_REL_I386_DIR32 .text
// COFF_I686-NEXT: {{.*}} IMAGE_REL_I386_SECREL .debug_frame
// COFF_I686-NEXT: {{.*}} IMAGE_REL_I386_DIR32 .text
