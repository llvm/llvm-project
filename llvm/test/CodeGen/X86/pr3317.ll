; RUN: llc < %s -march=x86
; PR3317

%VT = type [0 x i32 (...)*]
        %ArraySInt16 = type { %JavaObject, i8*, [0 x i16] }
        %ArraySInt8 = type { %JavaObject, i8*, [0 x i8] }
        %Attribut = type { %ArraySInt16*, i32, i32 }
        %CacheNode = type { i8*, %JavaCommonClass*, %CacheNode*, %Enveloppe* }
        %Enveloppe = type { %CacheNode*, %ArraySInt16*, %ArraySInt16*, i8, %JavaClass*, %CacheNode }
        %JavaArray = type { %JavaObject, i8* }
        %JavaClass = type { %JavaCommonClass, i32, %VT*, [1 x %TaskClassMirror], i8*, %JavaField*, i16, %JavaField*, i16, %JavaMethod*, i16, %JavaMethod*, i16, i8*, %ArraySInt8*, i8*, %Attribut*, i16, %JavaClass**, i16, %JavaClass*, i16, i8, i32, i32, i8*, void (i8*)* }
        %JavaCommonClass = type { %JavaCommonClass**, i32, [1 x %JavaObject*], i16, %JavaClass**, i16, %ArraySInt16*, %JavaClass*, i8* }
        %JavaField = type { i8*, i16, %ArraySInt16*, %ArraySInt16*, %Attribut*, i16, %JavaClass*, i32, i16, i8* }
        %JavaMethod = type { i8*, i16, %Attribut*, i16, %Enveloppe*, i16, %JavaClass*, %ArraySInt16*, %ArraySInt16*, i8, i8*, i32, i8* }
        %JavaObject = type { %VT*, %JavaCommonClass*, i8* }
        %TaskClassMirror = type { i32, i8* }
        %UTF8 = type { %JavaObject, i8*, [0 x i16] }

declare void @jnjvmNullPointerException()

define i32 @JnJVM_java_rmi_activation_ActivationGroupID_hashCode__(%JavaObject* nocapture) nounwind {
start:
        %1 = getelementptr %JavaObject, %JavaObject* %0, i64 1, i32 1                ; <%JavaCommonClass**> [#uses=1]
        %2 = load %JavaCommonClass*, %JavaCommonClass** %1         ; <%JavaCommonClass*> [#uses=4]
        %3 = icmp eq %JavaCommonClass* %2, null         ; <i1> [#uses=1]
        br i1 %3, label %verifyNullExit1, label %verifyNullCont2

verifyNullExit1:                ; preds = %start
        tail call void @jnjvmNullPointerException()
        unreachable

verifyNullCont2:                ; preds = %start
        %4 = bitcast %JavaCommonClass* %2 to { %JavaObject, i16, i32, i64 }*            ; <{ %JavaObject, i16, i32, i64 }*> [#uses=1]
        %5 = getelementptr { %JavaObject, i16, i32, i64 }, { %JavaObject, i16, i32, i64 }* %4, i64 0, i32 2             ; <i32*> [#uses=1]
        %6 = load i32, i32* %5               ; <i32> [#uses=1]
        %7 = getelementptr %JavaCommonClass, %JavaCommonClass* %2, i64 0, i32 4           ; <%JavaClass***> [#uses=1]
        %8 = bitcast %JavaClass*** %7 to i64*           ; <i64*> [#uses=1]
        %9 = load i64, i64* %8               ; <i64> [#uses=1]
        %10 = trunc i64 %9 to i32               ; <i32> [#uses=1]
        %11 = getelementptr %JavaCommonClass, %JavaCommonClass* %2, i64 0, i32 3          ; <i16*> [#uses=1]
        %12 = load i16, i16* %11             ; <i16> [#uses=1]
        %13 = sext i16 %12 to i32               ; <i32> [#uses=1]
        %14 = xor i32 %10, %6           ; <i32> [#uses=1]
        %15 = xor i32 %14, %13          ; <i32> [#uses=1]
        ret i32 %15 
}
