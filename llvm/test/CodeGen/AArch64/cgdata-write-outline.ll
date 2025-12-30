; This test verifies whether an outlined function is encoded into the __llvm_outline section
; when the -codegen-data-generate flag is used.

; Verify whether an outlined function is always created, but only encoded into the section when the flag is used.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -filetype=obj %s -o %t_save
; RUN: llvm-objdump -d %t_save | FileCheck %s
; RUN: llvm-objdump -h %t_save | FileCheck %s --check-prefix=SECTNAME
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=false -filetype=obj %s -o %t_nosave
; RUN: llvm-objdump -d  %t_nosave | FileCheck %s
; RUN: llvm-objdump -h %t_nosave | FileCheck %s --check-prefix=NOSECTNAME

; CHECK: _OUTLINED_FUNCTION
; CHECK-NEXT:  mov
; CHECK-NEXT:  mov
; CHECK-NEXT:  b
; SECTNAME: __llvm_outline
; NOSECTNAME-NOT: __llvm_outline

; Verify the content of cgdata after it has been processed with llvm-cgdata.
; RUN: llvm-cgdata --merge %t_save -o %t_cgdata
; RUN: llvm-cgdata --convert %t_cgdata | FileCheck %s --check-prefix=TREE

; TREE: :outlined_hash_tree
; TREE: ---
; TREE-NEXT: 0:
; TREE-NEXT:   Hash:            0x0
; TREE-NEXT:   Terminals:       0
; TREE-NEXT:   SuccessorIds:    [ 1 ]
; TREE-NEXT: 1:
; TREE-NEXT:   Hash:            {{.}}
; TREE-NEXT:   Terminals:       0
; TREE-NEXT:   SuccessorIds:    [ 2 ]
; TREE-NEXT: 2:
; TREE-NEXT:   Hash:            {{.}}
; TREE-NEXT:   Terminals:       0
; TREE-NEXT:   SuccessorIds:    [ 3 ]
; TREE-NEXT: 3:
; TREE-NEXT:   Hash:            {{.}}
; TREE-NEXT:   Terminals:       2
; TREE-NEXT:   SuccessorIds:    [  ]
; TREE-NEXT: ...

declare i32 @g(i32, i32, i32)
define i32 @f1() minsize {
  %1 = call i32 @g(i32 10, i32 1, i32 2);
  ret i32 %1
}
define i32 @f2() minsize {
  %1 = call i32 @g(i32 20, i32 1, i32 2);
  ret i32 %1
}
