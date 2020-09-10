; RUN: opt < %s -basic-aa -memcpyopt -dse -S | FileCheck -enable-var-scope %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9"

%0 = type { x86_fp80, x86_fp80 }
%1 = type { i32, i32 }

define void @test1(%0* sret  %agg.result, x86_fp80 %z.0, x86_fp80 %z.1) nounwind  {
entry:
  %tmp2 = alloca %0
  %memtmp = alloca %0, align 16
  %tmp5 = fsub x86_fp80 0xK80000000000000000000, %z.1
  call void @ccoshl(%0* sret %memtmp, x86_fp80 %tmp5, x86_fp80 %z.0) nounwind
  %tmp219 = bitcast %0* %tmp2 to i8*
  %memtmp20 = bitcast %0* %memtmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %tmp219, i8* align 16 %memtmp20, i32 32, i1 false)
  %agg.result21 = bitcast %0* %agg.result to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %agg.result21, i8* align 16 %tmp219, i32 32, i1 false)
  ret void

; Check that one of the memcpy's are removed.
;; FIXME: PR 8643 We should be able to eliminate the last memcpy here.

; CHECK-LABEL: @test1(
; CHECK: call void @ccoshl
; CHECK: call void @llvm.memcpy
; CHECK-NOT: llvm.memcpy
; CHECK: ret void
}

declare void @ccoshl(%0* nocapture sret, x86_fp80, x86_fp80) nounwind


; The intermediate alloca and one of the memcpy's should be eliminated, the
; other should be related with a memmove.
define void @test2(i8* %P, i8* %Q) nounwind  {
  %memtmp = alloca %0, align 16
  %R = bitcast %0* %memtmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %R, i8* align 16 %P, i32 32, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %Q, i8* align 16 %R, i32 32, i1 false)
  ret void

; CHECK-LABEL: @test2(
; CHECK-NEXT: call void @llvm.memmove{{.*}}(i8* align 16 %Q, i8* align 16 %P
; CHECK-NEXT: ret void
}

; The intermediate alloca and one of the memcpy's should be eliminated, the
; other should be related with a memcpy.
define void @test2_memcpy(i8* noalias %P, i8* noalias %Q) nounwind  {
  %memtmp = alloca %0, align 16
  %R = bitcast %0* %memtmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %R, i8* align 16 %P, i32 32, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %Q, i8* align 16 %R, i32 32, i1 false)
  ret void

; CHECK-LABEL: @test2_memcpy(
; CHECK-NEXT: call void @llvm.memcpy{{.*}}(i8* align 16 %Q, i8* align 16 %P
; CHECK-NEXT: ret void
}




@x = external global %0

define void @test3(%0* noalias sret %agg.result) nounwind  {
  %x.0 = alloca %0
  %x.01 = bitcast %0* %x.0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %x.01, i8* align 16 bitcast (%0* @x to i8*), i32 32, i1 false)
  %agg.result2 = bitcast %0* %agg.result to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %agg.result2, i8* align 16 %x.01, i32 32, i1 false)
  ret void
; CHECK-LABEL: @test3(
; CHECK-NEXT: %x.0 = alloca
; CHECK-NEXT: %x.01 = bitcast
; CHECK-NEXT: %agg.result1 = bitcast
; CHECK-NEXT: call void @llvm.memcpy
; CHECK-NEXT: %agg.result2 = bitcast
; CHECK-NEXT: ret void
}


; PR8644
define void @test4(i8 *%P) {
  %A = alloca %1
  %a = bitcast %1* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %a, i8* align 4 %P, i64 8, i1 false)
  call void @test4a(i8* align 1 byval %a)
  ret void
; CHECK-LABEL: @test4(
; CHECK-NEXT: call void @test4a(
}

; Make sure we don't remove the memcpy if the source address space doesn't match the byval argument
define void @test4_addrspace(i8 addrspace(1)* %P) {
  %A = alloca %1
  %a = bitcast %1* %A to i8*
  call void @llvm.memcpy.p0i8.p1i8.i64(i8* align 4 %a, i8 addrspace(1)* align 4 %P, i64 8, i1 false)
  call void @test4a(i8* align 1 byval %a)
  ret void
; CHECK-LABEL: @test4_addrspace(
; CHECK: call void @llvm.memcpy.p0i8.p1i8.i64(
; CHECK-NEXT: call void @test4a(
}

declare void @test4a(i8* align 1 byval)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
declare void @llvm.memcpy.p0i8.p1i8.i64(i8* nocapture, i8 addrspace(1)* nocapture, i64, i1) nounwind
declare void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture, i64, i1) nounwind

%struct.S = type { i128, [4 x i8]}

@sS = external global %struct.S, align 16

declare void @test5a(%struct.S* align 16 byval) nounwind ssp


; rdar://8713376 - This memcpy can't be eliminated.
define i32 @test5(i32 %x) nounwind ssp {
entry:
  %y = alloca %struct.S, align 16
  %tmp = bitcast %struct.S* %y to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %tmp, i8* align 16 bitcast (%struct.S* @sS to i8*), i64 32, i1 false)
  %a = getelementptr %struct.S, %struct.S* %y, i64 0, i32 1, i64 0
  store i8 4, i8* %a
  call void @test5a(%struct.S* align 16 byval %y)
  ret i32 0
  ; CHECK-LABEL: @test5(
  ; CHECK: store i8 4
  ; CHECK: call void @test5a(%struct.S* byval align 16 %y)
}

;; Noop memcpy should be zapped.
define void @test6(i8 *%P) {
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %P, i8* align 4 %P, i64 8, i1 false)
  ret void
; CHECK-LABEL: @test6(
; CHECK-NEXT: ret void
}


; PR9794 - Should forward memcpy into byval argument even though the memcpy
; isn't itself 8 byte aligned.
%struct.p = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

define i32 @test7(%struct.p* nocapture align 8 byval %q) nounwind ssp {
entry:
  %agg.tmp = alloca %struct.p, align 4
  %tmp = bitcast %struct.p* %agg.tmp to i8*
  %tmp1 = bitcast %struct.p* %q to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %tmp, i8* align 4 %tmp1, i64 48, i1 false)
  %call = call i32 @g(%struct.p* align 8 byval %agg.tmp) nounwind
  ret i32 %call
; CHECK-LABEL: @test7(
; CHECK: call i32 @g(%struct.p* byval align 8 %q) [[$NUW:#[0-9]+]]
}

declare i32 @g(%struct.p* align 8 byval)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

; PR11142 - When looking for a memcpy-memcpy dependency, don't get stuck on
; instructions between the memcpy's that only affect the destination pointer.
@test8.str = internal constant [7 x i8] c"ABCDEF\00"

define void @test8() {
; CHECK: test8
; CHECK-NOT: memcpy
  %A = tail call i8* @malloc(i32 10)
  %B = getelementptr inbounds i8, i8* %A, i64 2
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %B, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @test8.str, i64 0, i64 0), i32 7, i1 false)
  %C = tail call i8* @malloc(i32 10)
  %D = getelementptr inbounds i8, i8* %C, i64 2
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %D, i8* %B, i32 7, i1 false)
  ret void
; CHECK: ret void
}

declare noalias i8* @malloc(i32)

; rdar://11341081
%struct.big = type { [50 x i32] }

define void @test9_addrspacecast() nounwind ssp uwtable {
entry:
; CHECK-LABEL: @test9_addrspacecast(
; CHECK: f1
; CHECK-NOT: memcpy
; CHECK: f2
  %b = alloca %struct.big, align 4
  %tmp = alloca %struct.big, align 4
  call void @f1(%struct.big* sret %tmp)
  %0 = addrspacecast %struct.big* %b to i8 addrspace(1)*
  %1 = addrspacecast %struct.big* %tmp to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* align 4 %0, i8 addrspace(1)* align 4 %1, i64 200, i1 false)
  call void @f2(%struct.big* %b)
  ret void
}

define void @test9() nounwind ssp uwtable {
entry:
; CHECK: test9
; CHECK: f1
; CHECK-NOT: memcpy
; CHECK: f2
  %b = alloca %struct.big, align 4
  %tmp = alloca %struct.big, align 4
  call void @f1(%struct.big* sret %tmp)
  %0 = bitcast %struct.big* %b to i8*
  %1 = bitcast %struct.big* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 %1, i64 200, i1 false)
  call void @f2(%struct.big* %b)
  ret void
}

; rdar://14073661.
; Test10 triggered assertion when the compiler try to get the size of the
; opaque type of *x, where the x is the formal argument with attribute 'sret'.

%opaque = type opaque
declare void @foo(i32* noalias nocapture)

define void @test10(%opaque* noalias nocapture sret %x, i32 %y) {
  %a = alloca i32, align 4
  store i32 %y, i32* %a
  call void @foo(i32* noalias nocapture %a)
  %c = load i32, i32* %a
  %d = bitcast %opaque* %x to i32*
  store i32 %c, i32* %d
  ret void
}

; don't create new addressspacecasts when we don't know they're safe for the target
define void @test11([20 x i32] addrspace(1)* nocapture dereferenceable(80) %P) {
  %A = alloca [20 x i32], align 4
  %a = bitcast [20 x i32]* %A to i8*
  %b = bitcast [20 x i32] addrspace(1)* %P to i8 addrspace(1)*
  call void @llvm.memset.p0i8.i64(i8* align 4 %a, i8 0, i64 80, i1 false)
  call void @llvm.memcpy.p1i8.p0i8.i64(i8 addrspace(1)* align 4 %b, i8* align 4 %a, i64 80, i1 false)
  ret void
; CHECK-LABEL: @test11(
; CHECK-NOT: addrspacecast
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
declare void @llvm.memcpy.p1i8.p0i8.i64(i8 addrspace(1)* nocapture, i8* nocapture, i64, i1) nounwind

declare void @f1(%struct.big* nocapture sret)
declare void @f2(%struct.big*)

; CHECK: attributes [[$NUW]] = { nounwind }
; CHECK: attributes #1 = { argmemonly nounwind willreturn }
; CHECK: attributes #2 = { nounwind ssp }
; CHECK: attributes #3 = { nounwind ssp uwtable }
