; RUN: llc < %s -march=bpfel -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=bpfeb -verify-machineinstrs | FileCheck %s

; Source code:
; int g[2];
;
; int test(ptr ctx) {
;   int a = 4, b;
;   unsigned long long c = 333333333333ULL;
;   asm volatile("r0 = *(u16 *)skb[%0]" : : "i"(2));
;   asm volatile("r0 = *(u16 *)skb[%0]" : : "r"(a));
;   asm volatile("%0 = %1" : "=r"(b) : "i"(4));
;   asm volatile("%0 = %1 ll" : "=r"(b) : "i"(c));
;   asm volatile("%0 = *(u16 *) %1" : "=r"(b) : "m"(a));
;   asm volatile("%0 = *(u32 *) %1" : "=r"(b) : "m"(g[1]));
;   return b;
; }
;

@g = common global [2 x i32] zeroinitializer, align 4

; Function Attrs: nounwind
define i32 @test(ptr nocapture readnone %ctx) local_unnamed_addr #0 {
entry:
  %a = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %a) #2
  store i32 4, ptr %a, align 4
  tail call void asm sideeffect "r0 = *(u16 *)skb[$0]", "i"(i32 2) #2
; CHECK: r0 = *(u16 *)skb[2]
  tail call void asm sideeffect "r0 = *(u16 *)skb[$0]", "r"(i32 4) #2
; CHECK: r0 = *(u16 *)skb[r1]
  %0 = tail call i32 asm sideeffect "$0 = $1", "=r,i"(i32 4) #2
; CHECK: r1 = 4
  %1 = tail call i32 asm sideeffect "$0 = $1 ll", "=r,i"(i64 333333333333) #2
; CHECK: r1 = 333333333333 ll
  %2 = call i32 asm sideeffect "$0 = *(u16 *) $1", "=r,*m"(ptr elementtype(i32) nonnull %a) #2
; CHECK: r1 = *(u16 *)(r10 - 4)
  %3 = call i32 asm sideeffect "$0 = *(u32 *) $1", "=r,*m"(ptr elementtype(i32) getelementptr inbounds ([2 x i32], ptr @g, i64 0, i64 1)) #2
; CHECK: r1 = g ll
; CHECK: r0 = *(u32 *)(r1 + 4)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %a) #2
  ret i32 %3
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }
