; RUN: llvm-as < %s | llc -march=x86 | grep lea

	%struct.anon = type { [3 x double], double, %struct.node*, [64 x %struct.bnode*], [64 x %struct.bnode*] }
	%struct.bnode = type { i16, double, [3 x double], i32, i32, [3 x double], [3 x double], [3 x double], double, %struct.bnode*, %struct.bnode* }
	%struct.node = type { i16, double, [3 x double], i32, i32 }

define fastcc void @old_main() {
entry:
	%tmp44 = malloc %struct.anon		; <%struct.anon*> [#uses=2]
	store double 4.000000e+00, double* null, align 4
	br label %bb41

bb41:		; preds = %uniform_testdata.exit, %entry
	%i.0110 = phi i32 [ 0, %entry ], [ %tmp48, %uniform_testdata.exit ]		; <i32> [#uses=2]
	%tmp48 = add i32 %i.0110, 1		; <i32> [#uses=1]
	br i1 false, label %uniform_testdata.exit, label %bb33.preheader.i

bb33.preheader.i:		; preds = %bb41
	ret void

uniform_testdata.exit:		; preds = %bb41
	%tmp57 = getelementptr %struct.anon* %tmp44, i32 0, i32 3, i32 %i.0110		; <%struct.bnode**> [#uses=1]
	store %struct.bnode* null, %struct.bnode** %tmp57, align 4
	br i1 false, label %bb154, label %bb41

bb154:		; preds = %bb154, %uniform_testdata.exit
	br i1 false, label %bb166, label %bb154

bb166:		; preds = %bb154
	%tmp169 = getelementptr %struct.anon* %tmp44, i32 0, i32 3, i32 0		; <%struct.bnode**> [#uses=0]
	ret void
}
