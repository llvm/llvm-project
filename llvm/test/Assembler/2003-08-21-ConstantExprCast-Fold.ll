; RUN: llvm-as < %s | llvm-dis | not grep getelementptr
; RUN: verify-uselistorder %s

@A = external global { float }          ; <ptr> [#uses=2]
@0 = global ptr @A             ; <ptr>:0 [#uses=0]
