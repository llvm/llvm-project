; RUN: llc -o - %s -mtriple=aarch64-windows | FileCheck %s
; Check that we allocate the unwind help stack object in a fixed location from fp
; so that the runtime can find it when handling an exception
target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-pc-windows-msvc19.25.28611"

; Check that the store to the unwind help object for func2 is via FP
; CHECK-LABEL: ?func2@@YAXXZ
; CHECK: mov x[[#SCRATCH_REG:]], #-2
; CHECK: stur x[[#SCRATCH_REG:]], [x29, #[[#]]]
;
; // struct that requires greater than stack alignment
; struct alignas(32) A
; {
;     // data that would be invalid for unwind help (> 0)
;     int _x[4]{42, 42, 42, 42};
;     ~A() {}
; };
; 
; // cause us to run the funclet in func2
; void func3()
; {
;     throw 1;
; }
; 
; // the funclet that ensures we have the unwind help correct
; void func2()
; {
;     A a;
;     func3();
; }
; 
; // function to ensure we are misaligned in func2
; void func1()
; {
;     func2();
; }
; 
; // set things up and ensure alignment for func1
; void test()
; {
;     try {
;         A a;
;         func1();
;     } catch(...) {}
; }

%struct.A = type { [4 x i32], [16 x i8] }
declare dso_local ptr @"??0A@@QEAA@XZ"(ptr returned %0)
declare dso_local void @"??1A@@QEAA@XZ"(ptr %0)
declare dso_local i32 @__CxxFrameHandler3(...)
declare dso_local void @"?func3@@YAXXZ"()

; Function Attrs: noinline optnone uwtable
define dso_local void @"?func2@@YAXXZ"() #0 personality ptr @__CxxFrameHandler3 {
  %1 = alloca %struct.A, align 32
  %2 = call ptr @"??0A@@QEAA@XZ"(ptr %1) #3
  invoke void @"?func3@@YAXXZ"()
          to label %3 unwind label %4

3:                                                ; preds = %0
  call void @"??1A@@QEAA@XZ"(ptr %1) #3
  ret void

4:                                                ; preds = %0
  %5 = cleanuppad within none []
  call void @"??1A@@QEAA@XZ"(ptr %1) #3 [ "funclet"(token %5) ]
  cleanupret from %5 unwind to caller
}
