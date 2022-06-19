; RUN: opt < %s -passes=ipsccp -S | FileCheck %s
; Don't constant-propagate byval pointers, since they are not pointers!
; PR5038
%struct.MYstr = type { i8, i32 }
@mystr = internal global %struct.MYstr zeroinitializer ; <%struct.MYstr*> [#uses=3]
define internal void @vfu1(%struct.MYstr* byval(%struct.MYstr) align 4 %u) nounwind {
entry:
  %0 = getelementptr %struct.MYstr, %struct.MYstr* %u, i32 0, i32 1 ; <i32*> [#uses=1]
  store i32 99, i32* %0, align 4
; CHECK: %struct.MYstr* %u
  %1 = getelementptr %struct.MYstr, %struct.MYstr* %u, i32 0, i32 0 ; <i8*> [#uses=1]
  store i8 97, i8* %1, align 4
; CHECK: %struct.MYstr* %u
  br label %return

return:                                           ; preds = %entry
  ret void
}

define internal i32 @vfu2(%struct.MYstr* byval(%struct.MYstr) align 4 %u) nounwind readonly {
entry:
  %0 = getelementptr %struct.MYstr, %struct.MYstr* %u, i32 0, i32 1 ; <i32*> [#uses=1]
  %1 = load i32, i32* %0
; CHECK: %struct.MYstr* @mystr, i{{..}} 0, i32 1
  %2 = getelementptr %struct.MYstr, %struct.MYstr* %u, i32 0, i32 0 ; <i8*> [#uses=1]
  %3 = load i8, i8* %2
; CHECK: %struct.MYstr* @mystr, i{{..}} 0, i32 0
  %4 = zext i8 %3 to i32
  %5 = add i32 %4, %1
  ret i32 %5
}

define i32 @unions() nounwind {
entry:
  call void @vfu1(%struct.MYstr* byval(%struct.MYstr) align 4 @mystr) nounwind
  %result = call i32 @vfu2(%struct.MYstr* byval(%struct.MYstr) align 4 @mystr) nounwind
; CHECK: ret i32 %result
  ret i32 %result
}

