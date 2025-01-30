; The CGSCC pass manager includes an SCC iteration utility that tracks indirect
; calls that are turned into direct calls (devirtualization) and re-visits the
; SCC to expose those calls to the SCC-based IPO passes. We trigger
; devirtualization here with GVN which forwards a store through a load and to
; an indirect call.
;
; RUN: opt -aa-pipeline=basic-aa -passes='module(inferattrs),cgscc(function-attrs,function(gvn,instcombine))' -S < %s | FileCheck %s --check-prefix=CHECK --check-prefix=BEFORE
; RUN: opt -aa-pipeline=basic-aa -passes='module(inferattrs),cgscc(devirt<1>(function-attrs,function(gvn,instcombine)))' -S < %s | FileCheck %s --check-prefix=CHECK --check-prefix=AFTER --check-prefix=AFTER1
; RUN: opt -aa-pipeline=basic-aa -passes='module(inferattrs),cgscc(devirt<2>(function-attrs,function(gvn,instcombine)))' -S < %s | FileCheck %s --check-prefix=CHECK --check-prefix=AFTER --check-prefix=AFTER2
;
; RUN: not --crash opt -abort-on-max-devirt-iterations-reached -aa-pipeline=basic-aa -passes='module(inferattrs),cgscc(devirt<1>(function-attrs,function(gvn,instcombine)))' -S < %s
; RUN: opt -abort-on-max-devirt-iterations-reached -aa-pipeline=basic-aa -passes='module(inferattrs),cgscc(devirt<2>(function-attrs,function(gvn,instcombine)))' -S < %s
;
; We also verify that the real O2 pipeline catches these cases.
; RUN: opt -aa-pipeline=basic-aa -passes='default<O2>' -S < %s | FileCheck %s --check-prefix=CHECK --check-prefix=AFTER --check-prefix=AFTER2

declare void @readnone() readnone
; CHECK: Function Attrs: nofree nosync memory(none)
; CHECK-NEXT: declare void @readnone()

declare void @unknown()
; CHECK-NOT: Function Attrs
; CHECK-LABEL: declare void @unknown(){{ *$}}

; The @test1 function checks that when we refine an indirect call to a direct
; call we revisit the SCC passes to reflect the more precise information. This
; is the basic functionality.

define void @test1() {
; BEFORE-NOT: Function Attrs
; AFTER: Function Attrs: nofree nosync memory(none)
; CHECK-LABEL: define void @test1()
entry:
  %fptr = alloca ptr
  store ptr @readnone, ptr %fptr
  %f = load ptr, ptr %fptr
  call void %f()
  ret void
}

; The @test2_* functions check that when we need multiple (in this case 2)
; repetitions to compute some state that is incrementally exposed with each
; one, the limit on repetitions is enforced. So we make progress with
; one repetition but not as much as with three.
;
; This is somewhat awkward to test because we have to contrive to have a state
; repetition triggered and observed with very few passes. The technique here
; is to have one indirect call that can only be resolved when the entire SCC is
; deduced as readonly, and mark that indirect call at the call site as readonly
; to make that possible. This forces us to first deduce readonly, then
; devirtualize again, and then deduce readnone.

declare void @readnone_with_arg(ptr) readnone
; CHECK: Function Attrs: nofree nosync memory(none)
; CHECK-LABEL: declare void @readnone_with_arg(ptr)

define void @test2_a(ptr %ignore) {
; BEFORE-NOT: Function Attrs
; AFTER1: Function Attrs: nofree memory(read)
; AFTER2: Function Attrs: nofree nosync memory(none)
; BEFORE: define void @test2_a(ptr %ignore)
; AFTER: define void @test2_a(ptr readnone %ignore)
entry:
  %f1ptr = alloca ptr
  store ptr @readnone_with_arg, ptr %f1ptr
  %f1 = load ptr, ptr %f1ptr
  ; This indirect call is the first to be resolved, allowing us to deduce
  ; readonly but not (yet) readnone.
  call void %f1(ptr %ignore)
; CHECK: call void @readnone_with_arg(ptr %ignore)

  ; Bogus call to test2_b to make this a cycle.
  call void @test2_b()

  ret void
}

define void @test2_b() {
; BEFORE-NOT: Function Attrs
; AFTER1: Function Attrs: nofree memory(read)
; AFTER2: Function Attrs: nofree nosync memory(none)
; CHECK-LABEL: define void @test2_b()
entry:
  %f2ptr = alloca ptr
  store ptr @readnone, ptr %f2ptr
  ; Call the other function here to prevent forwarding until the SCC has had
  ; function attrs deduced.
  call void @test2_a(ptr %f2ptr)

  %f2 = load ptr, ptr %f2ptr
  ; This is the second indirect call to be resolved, and can only be resolved
  ; after we deduce 'readonly' for the rest of the SCC. Once it is
  ; devirtualized, we can deduce readnone for the SCC.
  call void %f2() readonly
; BEFORE: call void %f2()
; AFTER: call void @readnone()

  ret void
}

declare ptr @memcpy(ptr, ptr, i64)
; CHECK-LABEL: ptr @memcpy(

; The @test3 function checks that when we refine an indirect call to an
; intrinsic we still revisit the SCC pass. This also covers cases where the
; value handle itself doesn't persist due to the nature of how instcombine
; creates the memcpy intrinsic call, and we rely on the count of indirect calls
; decreasing and the count of direct calls increasing.
; Adding 'noinline' attribute to force attributes for improved matching.
define void @test3(ptr %src, ptr %dest, i64 %size) noinline {
; CHECK: Function Attrs
; CHECK-NOT: read
; CHECK-SAME: noinline
; BEFORE-LABEL: define void @test3(ptr %src, ptr %dest, i64 %size)
; AFTER-LABEL: define void @test3(ptr nocapture readonly %src, ptr nocapture writeonly %dest, i64 %size)
  %fptr = alloca ptr
  store ptr @memcpy, ptr %fptr
  %f = load ptr, ptr %fptr
  call ptr %f(ptr %dest, ptr %src, i64 %size)
; CHECK: call void @llvm.memcpy
  ret void
}

; A boring function that just keeps our declarations around.
define void @keep(ptr %sink) {
; CHECK-NOT: Function Attrs
; CHECK-LABEL: define void @keep(
entry:
  store volatile ptr @readnone, ptr %sink
  store volatile ptr @unknown, ptr %sink
  store volatile ptr @memcpy, ptr %sink
  call void @unknown()
  ret void
}
