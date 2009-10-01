; RUN: llvm-as <%s | opt -ipsccp | llvm-dis | FileCheck %s
; Don't constant-propagate byval pointers, since they are not pointers!
; PR5038
%struct.MYstr = type { i8, i32 }
@mystr = internal global %struct.MYstr zeroinitializer ; <%struct.MYstr*> [#uses=3]
define internal void @vfu1(%struct.MYstr* byval align 4 %u) nounwind {
entry:
  %0 = getelementptr %struct.MYstr* %u, i32 0, i32 1 ; <i32*> [#uses=1]
  store i32 99, i32* %0, align 4
; CHECK: %struct.MYstr* %u
  %1 = getelementptr %struct.MYstr* %u, i32 0, i32 0 ; <i8*> [#uses=1]
  store i8 97, i8* %1, align 4
; CHECK: %struct.MYstr* %u
  br label %return

return:                                           ; preds = %entry
  ret void
}
define void @unions() nounwind {
entry:
  call void @vfu1(%struct.MYstr* byval align 4 @mystr) nounwind
  ret void
}

