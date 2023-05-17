; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

%struct.foobar = type { i32 }

@bar.d = internal unnamed_addr constant %struct.foobar zeroinitializer, align 4
@foo.d = internal constant %struct.foobar zeroinitializer, align 4

define i32 @main() unnamed_addr nounwind ssp {
entry:
  %call2 = tail call i32 @zed(ptr @foo.d, ptr @bar.d) nounwind
  ret i32 0
}

declare i32 @zed(ptr, ptr)

; CHECK: @bar.d = internal unnamed_addr constant %struct.foobar zeroinitializer, align 4
; CHECK: @foo.d = internal constant %struct.foobar zeroinitializer, align 4
; CHECK: define i32 @main() unnamed_addr #0 {

; CHECK: attributes #0 = { nounwind ssp }
