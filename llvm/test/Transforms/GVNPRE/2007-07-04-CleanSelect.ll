; RUN: llvm-as < %s | opt -gvnpre | llvm-dis

define i32* @_ZN6Solver9propagateEv(i32* %this) {
entry:
	%tmp15.i48 = load i8* null		; <i8> [#uses=2]
	%tmp64.i.i51 = sub i8 0, %tmp15.i48		; <i8> [#uses=1]
	%tmp231.i52 = select i1 false, i8 %tmp15.i48, i8 %tmp64.i.i51		; <i8> [#uses=0]
	ret i32* null
}
