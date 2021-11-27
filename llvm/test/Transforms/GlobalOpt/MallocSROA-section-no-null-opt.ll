; RUN: opt -passes=globalopt -S < %s | FileCheck %s
; CHECK: @Y
; CHECK: section ".foo"

%struct.xyz = type { double, i32 }

@Y = internal global %struct.xyz* null ,section ".foo"            ; <%struct.xyz**> [#uses=2]
@numf2s = external global i32                     ; <i32*> [#uses=1]

define void @init_net() #0 {
; CHECK-LABEL: init_net(
; CHECK: load i32, i32* @numf2s
; CHECK: call i8* @malloc
; CHECK: store %struct.xyz* {{.*}}, %struct.xyz** @Y
entry:
  %0 = load i32, i32* @numf2s, align 4                 ; <i32> [#uses=1]
  %mallocsize2 = shl i32 %0, 4                    ; <i32> [#uses=1]
  %malloccall3 = tail call i8* @malloc(i32 %mallocsize2)  ; <i8*> [#uses=1]
  %1 = bitcast i8* %malloccall3 to %struct.xyz*   ; <%struct.xyz*> [#uses=1]
  store %struct.xyz* %1, %struct.xyz** @Y, align 8
  ret void
}

define %struct.xyz* @load_train() #0 {
; CHECK-LABEL: load_train(
; CHECK: load %struct.xyz*, %struct.xyz** @Y
entry:
  %0 = load %struct.xyz*, %struct.xyz** @Y, align 8             ; <%struct.xyz*> [#uses=0]
  ret %struct.xyz* %0
}

declare noalias i8* @malloc(i32)

attributes #0 = { null_pointer_is_valid }
