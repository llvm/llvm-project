; RUN: llc < %s -mtriple=i686--
; PR3317

%VT = type [0 x ptr]
        %ArraySInt16 = type { %JavaObject, ptr, [0 x i16] }
        %ArraySInt8 = type { %JavaObject, ptr, [0 x i8] }
        %Attribut = type { ptr, i32, i32 }
        %CacheNode = type { ptr, ptr, ptr, ptr }
        %Enveloppe = type { ptr, ptr, ptr, i8, ptr, %CacheNode }
        %JavaArray = type { %JavaObject, ptr }
        %JavaClass = type { %JavaCommonClass, i32, ptr, [1 x %TaskClassMirror], ptr, ptr, i16, ptr, i16, ptr, i16, ptr, i16, ptr, ptr, ptr, ptr, i16, ptr, i16, ptr, i16, i8, i32, i32, ptr, ptr }
        %JavaCommonClass = type { ptr, i32, [1 x ptr], i16, ptr, i16, ptr, ptr, ptr }
        %JavaField = type { ptr, i16, ptr, ptr, ptr, i16, ptr, i32, i16, ptr }
        %JavaMethod = type { ptr, i16, ptr, i16, ptr, i16, ptr, ptr, ptr, i8, ptr, i32, ptr }
        %JavaObject = type { ptr, ptr, ptr }
        %TaskClassMirror = type { i32, ptr }
        %UTF8 = type { %JavaObject, ptr, [0 x i16] }

declare void @jnjvmNullPointerException()

define i32 @JnJVM_java_rmi_activation_ActivationGroupID_hashCode__(ptr nocapture) nounwind {
start:
        %1 = getelementptr %JavaObject, ptr %0, i64 1, i32 1                ; <ptr> [#uses=1]
        %2 = load ptr, ptr %1         ; <ptr> [#uses=4]
        %3 = icmp eq ptr %2, null         ; <i1> [#uses=1]
        br i1 %3, label %verifyNullExit1, label %verifyNullCont2

verifyNullExit1:                ; preds = %start
        tail call void @jnjvmNullPointerException()
        unreachable

verifyNullCont2:                ; preds = %start
        %4 = getelementptr { %JavaObject, i16, i32, i64 }, ptr %2, i64 0, i32 2             ; <ptr> [#uses=1]
        %5 = load i32, ptr %4               ; <i32> [#uses=1]
        %6 = getelementptr %JavaCommonClass, ptr %2, i64 0, i32 4           ; <ptr> [#uses=1]
        %7 = load i64, ptr %6               ; <i64> [#uses=1]
        %8 = trunc i64 %7 to i32               ; <i32> [#uses=1]
        %9 = getelementptr %JavaCommonClass, ptr %2, i64 0, i32 3          ; <ptr> [#uses=1]
        %10 = load i16, ptr %9             ; <i16> [#uses=1]
        %11 = sext i16 %10 to i32               ; <i32> [#uses=1]
        %12 = xor i32 %8, %5           ; <i32> [#uses=1]
        %13 = xor i32 %12, %11          ; <i32> [#uses=1]
        ret i32 %13 
}
