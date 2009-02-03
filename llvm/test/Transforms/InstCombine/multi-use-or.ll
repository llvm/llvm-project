; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {add double .sx, .sy}
; The 'or' has multiple uses, make sure that this doesn't prevent instcombine
; from propagating the extends to the truncs.

define double @ScaleObjectAdd(double %sx, double %sy, double %sz) nounwind {
entry:
        %sx34 = bitcast double %sx to i64               ; <i64> [#uses=1]
        %sx3435 = zext i64 %sx34 to i192                ; <i192> [#uses=1]
        %sy22 = bitcast double %sy to i64               ; <i64> [#uses=1]
        %sy2223 = zext i64 %sy22 to i192                ; <i192> [#uses=1]
        %sy222324 = shl i192 %sy2223, 128               ; <i192> [#uses=1]
        %sy222324.ins = or i192 %sx3435, %sy222324              ; <i192> [#uses=1]
        
        
        %a = trunc i192 %sy222324.ins to i64            ; <i64> [#uses=1]
        %b = bitcast i64 %a to double           ; <double> [#uses=1]
        %c = lshr i192 %sy222324.ins, 128               ; <i192> [#uses=1]
        %d = trunc i192 %c to i64               ; <i64> [#uses=1]
        %e = bitcast i64 %d to double           ; <double> [#uses=1]
        %f = add double %b, %e

;        ret double %e
        ret double %f
}
