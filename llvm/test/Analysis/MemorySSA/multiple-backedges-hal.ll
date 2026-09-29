; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s

; hfinkel's case
; [entry]
;  |
; .....
; (clobbering access - b)
;  |
; ....  ________________________________
;  \   /                               |
;   (x)                                |
;  ......                              |
;    |                                 |
;    |    ______________________       |
;     \   /                    |       |
; (starting access)            |       |
;     ...                      |       |
; (clobbering access - a)      |       |
;    ...                       |       |
;    | |                       |       |
;    | |_______________________|       |
;    |                                 |
;    |_________________________________|
;
; More specifically, one access, with multiple clobbering accesses. One of
; which strictly dominates the access, the other of which has a backedge

; readnone so we don't have a 1:1 mapping of MemorySSA edges to Instructions.
declare void @doThingWithoutReading() readnone
declare i8 @getValue() readnone
declare i1 @getBool() readnone

define hidden void @testcase(ptr %Arg) {
Entry:
  call void @doThingWithoutReading()
  %Val.Entry = call i8 @getValue()
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 %Val.Entry
  store i8 %Val.Entry, ptr %Arg
  call void @doThingWithoutReading()
  br label %OuterLoop

OuterLoop:
; CHECK: 5 = MemoryPhi({Entry,1},{InnerLoop.Tail,3})
; CHECK-NEXT: %Val.Outer =
  %Val.Outer = call i8 @getValue()
; CHECK: 2 = MemoryDef(5)
; CHECK-NEXT: store i8 %Val.Outer
  store i8 %Val.Outer, ptr %Arg
  call void @doThingWithoutReading()
  br label %InnerLoop

InnerLoop:
; CHECK: 4 = MemoryPhi({OuterLoop,2},{InnerLoop,3})
; CHECK-NEXT: ; MemoryUse(4)
; CHECK-NEXT: %StartingAccess = load
  %StartingAccess = load i8, ptr %Arg, align 4
  %Val.Inner = call i8 @getValue()
; CHECK: 3 = MemoryDef(4)
; CHECK-NEXT: store i8 %Val.Inner
  store i8 %Val.Inner, ptr %Arg
  call void @doThingWithoutReading()
  %KeepGoing = call i1 @getBool()
  br i1 %KeepGoing, label %InnerLoop.Tail, label %InnerLoop

InnerLoop.Tail:
  %KeepGoing.Tail = call i1 @getBool()
  br i1 %KeepGoing.Tail, label %End, label %OuterLoop

End:
  ret void
}
