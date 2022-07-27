; RUN: opt < %s -early-cse-memssa -earlycse-debug-hash -verify-memoryssa -disable-output
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Tests below highlight scenarios where EarlyCSE does not preserve MemorySSA
; optimized accesses. Current MemorySSA verify will accept these.

; Test 1:
; AA cannot tell here that the last load does not alias the only store.
; The first two loads are a common expression, EarlyCSE removes the second one,
; and then AA can see that the last load is a Use(LoE). Hence not optimized as
; it claims. Note that if we replace the GEP indices 2 and 1, AA sees NoAlias
; for the last load, before CSE-ing the first 2 loads.
%struct.ImageParameters = type { i32, i32, i32 }
@img = external global ptr, align 8
define void @test1_macroblock() {
entry:
  ; MemoryUse(LoE)
  %0 = load ptr, ptr @img, align 8

  %Pos_2 = getelementptr inbounds %struct.ImageParameters, ptr %0, i64 0, i32 2
  ; 1 = MemoryDef(LoE)
  store i32 undef, ptr %Pos_2, align 8

  ; MemoryUse(LoE)
  %1 = load ptr, ptr @img, align 8

  %Pos_1 = getelementptr inbounds %struct.ImageParameters, ptr %1, i64 0, i32 1
  ; MemoryUse(1) MayAlias
  %2 = load i32, ptr %Pos_1, align 4
  unreachable
}

; Test 2:
; EarlyCSE simplifies %string to undef. Def and Use used to be MustAlias, with
; undef they are NoAlias. The Use can be optimized further to LoE. We can
; de-optimize uses of replaced instructions, but in general this is not enough
; (see next tests).
%struct.TermS = type { i32, i32, i32, i32, i32, ptr }
define fastcc void @test2_term_string() {
entry:
  %string = getelementptr inbounds %struct.TermS, ptr undef, i64 0, i32 5
  ; 1 = MemoryDef(LoE)
  store ptr undef, ptr %string, align 8
  ; MemoryUse(1) MustAlias
  %0 = load ptr, ptr %string, align 8
  unreachable
}

; Test 3:
; EarlyCSE simplifies %0 to undef. So the second Def now stores to undef.
; We now find the second load (Use(2) can be optimized further to LoE)
; When replacing instructions, we can deoptimize all uses of the replaced
; instruction and all uses of transitive accesses. However this does not stop
; MemorySSA from being tripped by AA (see test4).
%struct.Grammar = type { ptr, ptr, %struct.anon }
%struct.anon = type { i32, i32, ptr, [3 x ptr] }
%struct.Term = type { i32 }

define fastcc void @test3_term_string(ptr %g) {
entry:
  ; 1 = MemoryDef(LoE)
  store ptr undef, ptr undef, align 8
  ; MemoryUse(LoE)
  %0 = load ptr, ptr undef, align 8
  %arrayidx = getelementptr inbounds i8, ptr %0, i64 undef
  ; 2 = MemoryDef(1)
  store i8 0, ptr %arrayidx, align 1
  %v = getelementptr inbounds %struct.Grammar, ptr %g, i64 0, i32 2, i32 2
  ; MemoryUse(2) MayAlias
  %1 = load ptr, ptr %v, align 8
  unreachable
}

; Test 4:
; Removing dead/unused instructions in if.then274 makes AA smarter. Before
; removal, it finds %4 MayAlias the store above. After removal this can be
; optimized to LoE. Hence after EarlyCSE, there is an access who claims is
; optimized and it can be optimized further.

; We can't escape such cases in general when relying on Alias Analysis.
; The only fail-safe way to actually preserve MemorySSA when removing or
; replacing instructions (i.e. get the *same* MemorySSA as if it was computed
; for the updated IR) is to recompute it from scratch. What we get now is still
; a correct update, but with accesses that claim to be optimized and can be
; optimized further if we were to re-run MemorySSA on the IR.
%struct.gnode.0.1.3.6.9.18.20.79 = type { i32, i32, i32, i32, i32, i32, i32, ptr }
@gnodeArray = external global ptr, align 8

define void @test4_shortest() {
entry:
  %exl.i = alloca [5 x i32], align 16
  br i1 undef, label %if.then274, label %for.cond404

if.then274:                                       ; preds = %if.end256
  %arrayidx.i = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 1
  %arrayidx1.i = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 2
  %arrayidx2.i = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 3
  %arrayidx3.i = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 4
  %arrayidx.i1034 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 1
  %arrayidx1.i1035 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 2
  %arrayidx2.i1036 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 3
  %arrayidx3.i1037 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 4
  unreachable

for.cond404:                                      ; preds = %if.end256
  %arrayidx.i960 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 1
  %arrayidx1.i961 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 2
  %arrayidx2.i962 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 3
  ; 1 = MemoryDef(LoE)
  store i32 undef, ptr %arrayidx2.i962, align 4
  %arrayidx3.i963 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 4

  ; MemoryUse(LoE)
  %0 = load ptr, ptr @gnodeArray, align 8
  %arrayidx6.i968 = getelementptr inbounds ptr, ptr %0, i64 undef
  ; MemoryUse(1) MayAlias
  %1 = load ptr, ptr %arrayidx6.i968, align 8
  br i1 undef, label %for.cond26.preheader.i974, label %if.then20.for.body_crit_edge.i999

for.cond26.preheader.i974:                        ; preds = %if.then20.i996
  %arrayidx.i924 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 1
  %arrayidx1.i925 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 2
  %arrayidx2.i926 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 3
  %arrayidx3.i927 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 4
  unreachable

if.then20.for.body_crit_edge.i999:                ; preds = %if.then20.i996
  %arrayidx9.phi.trans.insert.i997 = getelementptr inbounds [5 x i32], ptr %exl.i, i64 0, i64 undef
  unreachable
}
