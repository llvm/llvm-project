; RUN: opt < %s -passes=instcombine -S | grep "ret double .sy"

define internal double @ScaleObjectAdd(double %sx, double %sy, double %sz) nounwind {
entry:
        %sx34 = bitcast double %sx to i64               ; <i64> [#uses=1]
        %sx3435 = zext i64 %sx34 to i960                ; <i960> [#uses=1]
        %sy22 = bitcast double %sy to i64               ; <i64> [#uses=1]
        %sy2223 = zext i64 %sy22 to i960                ; <i960> [#uses=1]
        %sy222324 = shl i960 %sy2223, 320               ; <i960> [#uses=1]
        %sy222324.ins = or i960 %sx3435, %sy222324              ; <i960> [#uses=1]
        %sz10 = bitcast double %sz to i64               ; <i64> [#uses=1]
        %sz1011 = zext i64 %sz10 to i960                ; <i960> [#uses=1]
        %sz101112 = shl i960 %sz1011, 640               ; <i960> [#uses=1]
        %sz101112.ins = or i960 %sy222324.ins, %sz101112 
        
        %a = trunc i960 %sz101112.ins to i64            ; <i64> [#uses=1]
        %b = bitcast i64 %a to double           ; <double> [#uses=1]
        %c = lshr i960 %sz101112.ins, 320               ; <i960> [#uses=1]
        %d = trunc i960 %c to i64               ; <i64> [#uses=1]
        %e = bitcast i64 %d to double           ; <double> [#uses=1]
        %f = fadd double %b, %e

        ret double %e
}
