; RUN: llc < %s -mtriple=i386-apple-darwin9 -relocation-model=pic -frame-pointer=all | FileCheck %s



; PR7353 PR7334 rdar://8072315 rdar://8018308

define available_externally hidden 
void @_ZNSbIcED1Ev() nounwind readnone ssp align 2 {
entry:
  ret void
}

define ptr @test1() nounwind {
entry:
  ret ptr @_ZNSbIcED1Ev
}

; This must use movl of the stub, not an lea, since the function isn't being
; emitted here.
; CHECK: movl L__ZNSbIcED1Ev$non_lazy_ptr-L0$pb(




; <rdar://problem/7383328>

@.str = private constant [12 x i8] c"hello world\00", align 1 ; <ptr> [#uses=1]

define hidden void @func() nounwind ssp uwtable {
entry:
  %0 = call i32 @puts(ptr @.str) nounwind ; <i32> [#uses=0]
  br label %return

return:                                           ; preds = %entry
  ret void
}

declare i32 @puts(ptr)

define hidden i32 @main() nounwind ssp uwtable {
entry:
  %retval = alloca i32                            ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @func() nounwind
  br label %return

return:                                           ; preds = %entry
  %retval1 = load i32, ptr %retval                    ; <i32> [#uses=1]
  ret i32 %retval1
}
