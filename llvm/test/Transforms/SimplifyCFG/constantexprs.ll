; RUN: opt -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S < %s | FileCheck %s

; See that we do not crash when queriying cost model about the cost of constant expression extractelement.

%"struct.(anonymous namespace)::aj" = type { %struct.o }
%struct.o = type { %struct.j }
%struct.j = type { %struct.c }
%struct.c = type { %struct.a }
%struct.a = type { i8 }
%struct.e = type { %struct.a* }

$_ZN1eC2EPK1a = comdat any

@_ZN12_GLOBAL__N_12anE = internal global %"struct.(anonymous namespace)::aj" zeroinitializer, align 1

declare dso_local i32 @_Zeq1eS_(%struct.a*, %struct.a*) local_unnamed_addr #2

define internal fastcc %struct.a* @_ZNK1jIiN12_GLOBAL__N_12ajEE2aeERKi() unnamed_addr #0 align 2 {
; CHECK-LABEL: @_ZNK1jIiN12_GLOBAL__N_12ajEE2aeERKi
entry:
  %call = call i32 @_Zeq1eS_(%struct.a* null, %struct.a* null)
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %cond.false, label %cond.end

cond.false:                                       ; preds = %entry
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.false
  %retval.sroa.0.0 = phi %struct.a* [ null, %cond.false ], [ extractelement (<1 x %struct.a*> inttoptr (<1 x i64> bitcast (i64 ptrtoint (%"struct.(anonymous namespace)::aj"* @_ZN12_GLOBAL__N_12anE to i64) to <1 x i64>) to <1 x %struct.a*>), i64 0), %entry ]
  ret %struct.a*  %retval.sroa.0.0
}
