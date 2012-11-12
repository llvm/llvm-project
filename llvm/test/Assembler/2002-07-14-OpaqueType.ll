; Test that opaque types are preserved correctly
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis
;

%Ty = type opaque

define %Ty* @func() {
	ret %Ty* null
}
 
