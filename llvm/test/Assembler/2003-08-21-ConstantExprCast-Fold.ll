; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | not grep getelementptr
; RUN: verify-uselistorder --opaque-pointers=0 %s

@A = external global { float }          ; <{ float }*> [#uses=2]
@0 = global i32* bitcast ({ float }* @A to i32*)             ; <i32**>:0 [#uses=0]
