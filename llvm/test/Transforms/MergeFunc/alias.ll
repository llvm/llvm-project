; RUN: opt -S -passes=mergefunc -mergefunc-use-aliases < %s | FileCheck %s

; Aliases should always be created for the weak functions, and
; for external functions if there is no local function

; CHECK: @external_external_2 = unnamed_addr alias void (float*), bitcast (void (i32*)* @external_external_1 to void (float*)*)
; CHECK: @weak_weak_2 = weak unnamed_addr alias void (float*), bitcast (void (i32*)* @0 to void (float*)*)
; CHECK: @weak_weak_1 = weak unnamed_addr alias void (i32*), void (i32*)* @0
; CHECK: @weak_external_1 = weak unnamed_addr alias void (i32*), bitcast (void (float*)* @weak_external_2 to void (i32*)*)
; CHECK: @external_weak_2 = weak unnamed_addr alias void (float*), bitcast (void (i32*)* @external_weak_1 to void (float*)*)
; CHECK: @weak_internal_1 = weak unnamed_addr alias void (i32*), bitcast (void (float*)* @weak_internal_2 to void (i32*)*)
; CHECK: @internal_weak_2 = weak unnamed_addr alias void (float*), bitcast (void (i32*)* @internal_weak_1 to void (float*)*)

; A strong backing function had to be created for the weak-weak pair

; CHECK: define private void @0(i32* %a) unnamed_addr
; CHECK-NEXT: call void @dummy4()

; These internal functions are dropped in favor of the external ones

; CHECK-NOT: define internal void @external_internal_2(float *%a) unnamed_addr
; CHECK-NOT: define internal void @internal_external_1(i32 *%a) unnamed_addr
; CHECK-NOT: define internal void @internal_external_1(i32 *%a) unnamed_addr
; CHECK-NOT: define internal void @internal_external_2(float *%a) unnamed_addr

; Only used to mark which functions should be merged.
declare void @dummy1()
declare void @dummy2()
declare void @dummy3()
declare void @dummy4()
declare void @dummy5()
declare void @dummy6()
declare void @dummy7()
declare void @dummy8()
declare void @dummy9()

define void @external_external_1(i32 *%a) unnamed_addr {
  call void @dummy1()
  ret void
}
define void @external_external_2(float *%a) unnamed_addr {
  call void @dummy1()
  ret void
}

define void @external_internal_1(i32 *%a) unnamed_addr {
  call void @dummy2()
  ret void
}
define internal void @external_internal_2(float *%a) unnamed_addr {
  call void @dummy2()
  ret void
}

define internal void @internal_external_1(i32 *%a) unnamed_addr {
  call void @dummy3()
  ret void
}
define void @internal_external_2(float *%a) unnamed_addr {
  call void @dummy3()
  ret void
}

define weak void @weak_weak_1(i32 *%a) unnamed_addr {
  call void @dummy4()
  ret void
}
define weak void @weak_weak_2(float *%a) unnamed_addr {
  call void @dummy4()
  ret void
}

define weak void @weak_external_1(i32 *%a) unnamed_addr {
  call void @dummy5()
  ret void
}
define external void @weak_external_2(float *%a) unnamed_addr {
  call void @dummy5()
  ret void
}

define external void @external_weak_1(i32 *%a) unnamed_addr {
  call void @dummy6()
  ret void
}
define weak void @external_weak_2(float *%a) unnamed_addr {
  call void @dummy6()
  ret void
}

define weak void @weak_internal_1(i32 *%a) unnamed_addr {
  call void @dummy7()
  ret void
}
define internal void @weak_internal_2(float *%a) unnamed_addr {
  call void @dummy7()
  ret void
}

define internal void @internal_weak_1(i32 *%a) unnamed_addr {
  call void @dummy8()
  ret void
}
define weak void @internal_weak_2(float *%a) unnamed_addr {
  call void @dummy8()
  ret void
}

define internal void @internal_internal_1(i32 *%a) unnamed_addr {
  call void @dummy9()
  ret void
}
define internal void @internal_internal_2(float *%a) unnamed_addr {
  call void @dummy9()
  ret void
}
