; RUN: opt < %s -passes=lcssa

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"

@.str12 = external constant [3 x i8], align 1		; <ptr> [#uses=1]
@.str17175 = external constant [4 x i8], align 1		; <ptr> [#uses=1]
@.str21179 = external constant [12 x i8], align 1		; <ptr> [#uses=1]
@.str25183 = external constant [10 x i8], align 1		; <ptr> [#uses=1]
@.str32190 = external constant [92 x i8], align 1		; <ptr> [#uses=1]
@.str41 = external constant [25 x i8], align 1		; <ptr> [#uses=1]

define void @_ZN8EtherBus10initializeEv() personality ptr @__gxx_personality_v0 {
entry:
	br i1 undef, label %_ZN7cObjectnwEj.exit, label %bb.i

bb.i:		; preds = %entry
	br label %_ZN7cObjectnwEj.exit

_ZN7cObjectnwEj.exit:		; preds = %bb.i, %entry
	invoke void @_ZN7cObjectC2EPKc(ptr undef, ptr @.str21179)
			to label %bb1 unwind label %lpad

bb1:		; preds = %_ZN7cObjectnwEj.exit
	br i1 undef, label %_ZNK5cGate4sizeEv.exit, label %bb.i110

bb.i110:		; preds = %bb1
	br label %_ZNK5cGate4sizeEv.exit

_ZNK5cGate4sizeEv.exit:		; preds = %bb.i110, %bb1
	br i1 undef, label %_ZNK5cGate4sizeEv.exit122, label %bb.i120

bb.i120:		; preds = %_ZNK5cGate4sizeEv.exit
	br label %_ZNK5cGate4sizeEv.exit122

_ZNK5cGate4sizeEv.exit122:		; preds = %bb.i120, %_ZNK5cGate4sizeEv.exit
	br i1 undef, label %bb8, label %bb2

bb2:		; preds = %_ZNK5cGate4sizeEv.exit122
	unreachable

bb8:		; preds = %_ZNK5cGate4sizeEv.exit122
	%tmp = invoke ptr @_ZN7cModule3parEPKc(ptr undef, ptr @.str25183)
			to label %invcont9 unwind label %lpad119		; <ptr> [#uses=1]

invcont9:		; preds = %bb8
	%tmp1 = invoke ptr @_ZN4cPar11stringValueEv(ptr %tmp)
			to label %invcont10 unwind label %lpad119		; <ptr> [#uses=1]

invcont10:		; preds = %invcont9
	invoke void @_ZN8EtherBus8tokenizeEPKcRSt6vectorIdSaIdEE(ptr null, ptr %tmp1, ptr undef)
			to label %invcont11 unwind label %lpad119

invcont11:		; preds = %invcont10
	br i1 undef, label %bb12, label %bb18

bb12:		; preds = %invcont11
	invoke void (ptr, ptr, ...) @_ZN6cEnvir6printfEPKcz(ptr null, ptr @.str12, i32 undef)
			to label %bb.i.i159 unwind label %lpad119

bb.i.i159:		; preds = %bb12
	unreachable

bb18:		; preds = %invcont11
	br i1 undef, label %bb32, label %bb34

bb32:		; preds = %bb18
	br i1 undef, label %bb.i.i123, label %bb34

bb.i.i123:		; preds = %bb32
	br label %bb34

bb34:		; preds = %bb.i.i123, %bb32, %bb18
	%tmp2 = invoke ptr @_Znaj(i32 undef)
			to label %invcont35 unwind label %lpad119		; <ptr> [#uses=0]

invcont35:		; preds = %bb34
	br i1 undef, label %bb49, label %bb61

bb49:		; preds = %invcont35
	invoke void (ptr, ptr, ...) @_ZNK13cSimpleModule5errorEPKcz(ptr undef, ptr @.str32190)
			to label %bb51 unwind label %lpad119

bb51:		; preds = %bb49
	unreachable

bb61:		; preds = %invcont35
	br label %bb106

.noexc:		; preds = %bb106
	invoke void @_ZN7cObjectC2EPKc(ptr undef, ptr @.str41)
			to label %bb102 unwind label %lpad123

bb102:		; preds = %.noexc
	invoke void undef(ptr undef, i8 zeroext 1)
			to label %invcont103 unwind label %lpad119

invcont103:		; preds = %bb102
	invoke void undef(ptr undef, double 1.000000e+07)
			to label %invcont104 unwind label %lpad119

invcont104:		; preds = %invcont103
	%tmp3 = invoke i32 @_ZN13cSimpleModule11sendDelayedEP8cMessagedPKci(ptr undef, ptr undef, double 0.000000e+00, ptr @.str17175, i32 undef)
			to label %invcont105 unwind label %lpad119		; <i32> [#uses=0]

invcont105:		; preds = %invcont104
	br label %bb106

bb106:		; preds = %invcont105, %bb61
	%tmp4 = invoke ptr @_Znaj(i32 124)
			to label %.noexc unwind label %lpad119		; <ptr> [#uses=1]

lpad:		; preds = %_ZN7cObjectnwEj.exit
        %exn = landingpad {ptr, i32}
                 cleanup
	br label %Unwind

lpad119:		; preds = %bb106, %invcont104, %invcont103, %bb102, %bb49, %bb34, %bb12, %invcont10, %invcont9, %bb8
        %exn119 = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad123:		; preds = %.noexc
        %exn123 = landingpad {ptr, i32}
                 cleanup
	%tmp5 = icmp eq ptr %tmp4, null		; <i1> [#uses=1]
	br i1 %tmp5, label %Unwind, label %bb.i2

bb.i2:		; preds = %lpad123
	br label %Unwind

Unwind:		; preds = %bb.i2, %lpad123, %lpad
	unreachable
}

declare i32 @__gxx_personality_v0(...)

declare void @_ZN8EtherBus8tokenizeEPKcRSt6vectorIdSaIdEE(ptr nocapture, ptr, ptr)

declare ptr @_Znaj(i32)

declare void @_ZN6cEnvir6printfEPKcz(ptr nocapture, ptr nocapture, ...)

declare void @_ZNK13cSimpleModule5errorEPKcz(ptr nocapture, ptr nocapture, ...) noreturn

declare ptr @_ZN7cModule3parEPKc(ptr, ptr)

declare i32 @_ZN13cSimpleModule11sendDelayedEP8cMessagedPKci(ptr, ptr, double, ptr, i32)

declare void @_ZN7cObjectC2EPKc(ptr, ptr)

declare ptr @_ZN4cPar11stringValueEv(ptr)
