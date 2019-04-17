; RUN: opt -constmerge -S < %s | FileCheck %s
; PR8978

declare i32 @zed(%struct.foobar*, %struct.foobar*)

%struct.foobar = type { i32 }
; CHECK: bar.d
@bar.d =  unnamed_addr constant %struct.foobar zeroinitializer, align 4
; CHECK-NOT: foo.d
@foo.d = internal constant %struct.foobar zeroinitializer, align 4
define i32 @main() nounwind ssp {
entry:
; CHECK: bar.d
  %call2 = tail call i32 @zed(%struct.foobar* @foo.d, %struct.foobar* @bar.d)
nounwind
  ret i32 0
}

