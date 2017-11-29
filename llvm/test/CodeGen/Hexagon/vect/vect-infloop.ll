; Extracted from test/CodeGen/Generic/vector-casts.ll: used to loop indefinitely.
; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: combine

define void @a(<2 x double>* %p, <2 x i8>* %q) {
        %t = load <2 x double>, <2 x double>* %p
	%r = fptosi <2 x double> %t to <2 x i8>
        store <2 x i8> %r, <2 x i8>* %q
	ret void
}
