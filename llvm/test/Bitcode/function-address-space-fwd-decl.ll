; Verify that forward declarations from call instructions work even with non-zero AS
; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s

define void @call_named() {
entry:
  %0 = tail call addrspace(40) i32 @named(i16* null)
  ; CHECK: %0 = tail call addrspace(40) i32 @named(i16* null)
  ret void
}

define void @call_numbered() {
entry:
  %0 = tail call addrspace(40) i32 @0(i16* null)
  ; CHECK: %0 = tail call addrspace(40) i32 @0(i16* null)
  ret void
}


define i32 @invoked() personality i8* null {
entry:
  %0 = invoke addrspace(40) i32 @foo() to label %l1 unwind label %lpad
  ; CHECK: invoke addrspace(40) i32 @foo()
l1:
  br label %return
lpad:
  %1 = landingpad { i8*, i32 }
    catch i8* null
    catch i8* null
  ret i32 0
return:
  ret i32 0
}

declare i32 @foo() addrspace(40)
; CHECK: declare i32 @foo() addrspace(40)
declare i32 @named(i16* nocapture) addrspace(40)
; CHECK: declare i32 @named(i16* nocapture) addrspace(40)
declare i32 @0(i16*) addrspace(40)
; CHECK: declare i32 @0(i16*) addrspace(40)
