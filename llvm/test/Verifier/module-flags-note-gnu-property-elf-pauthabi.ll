; RUN: rm -rf %t && split-file %s %t && cd %t

; CHECK: either both or no 'aarch64-elf-pauthabi-platform' and 'aarch64-elf-pauthabi-version' module flags must be present

;--- err1.ll

; RUN: not llvm-as err1.ll -o /dev/null 2>&1 | FileCheck %s

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 2}

;--- err2.ll

; RUN: not llvm-as err2.ll -o /dev/null 2>&1 | FileCheck %s

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 31}
