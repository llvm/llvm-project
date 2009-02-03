; RUN: llvm-as < %s | opt -gvnpre | llvm-dis | grep tmp114115.gvnpre

	%struct.AV = type { %struct.XPVAV*, i32, i32 }
	%struct.CLONE_PARAMS = type { %struct.AV*, i32, %struct.PerlInterpreter* }
	%struct.HE = type { %struct.HE*, %struct.HEK*, %struct.SV* }
	%struct.HEK = type { i32, i32, [1 x i8] }
	%struct.HV = type { %struct.XPVHV*, i32, i32 }
	%struct.MAGIC = type { %struct.MAGIC*, %struct.MGVTBL*, i16, i8, i8, %struct.SV*, i8*, i32 }
	%struct.MGVTBL = type { i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*, %struct.SV*, i8*, i32)*, i32 (%struct.MAGIC*, %struct.CLONE_PARAMS*)* }
	%struct.OP = type { %struct.OP*, %struct.OP*, %struct.OP* ()*, i32, i16, i16, i8, i8 }
	%struct.PMOP = type { %struct.OP*, %struct.OP*, %struct.OP* ()*, i32, i16, i16, i8, i8, %struct.OP*, %struct.OP*, %struct.OP*, %struct.OP*, %struct.PMOP*, %struct.REGEXP*, i32, i32, i8, %struct.HV* }
	%struct.PerlInterpreter = type { i8 }
	%struct.REGEXP = type { i32*, i32*, %struct.regnode*, %struct.reg_substr_data*, i8*, %struct.reg_data*, i8*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, [1 x %struct.regnode] }
	%struct.SV = type { i8*, i32, i32 }
	%struct.XPVAV = type { i8*, i32, i32, i32, double, %struct.MAGIC*, %struct.HV*, %struct.SV**, %struct.SV*, i8 }
	%struct.XPVHV = type { i8*, i32, i32, i32, double, %struct.MAGIC*, %struct.HV*, i32, %struct.HE*, %struct.PMOP*, i8* }
	%struct.reg_data = type { i32, i8*, [1 x i8*] }
	%struct.reg_substr_data = type { [3 x %struct.reg_substr_datum] }
	%struct.reg_substr_datum = type { i32, i32, %struct.SV*, %struct.SV* }
	%struct.regnode = type { i8, i8, i16 }

define void @Perl_op_clear(%struct.OP* %o) {
entry:
	switch i32 0, label %bb106 [
		 i32 13, label %bb106
		 i32 31, label %clear_pmop
		 i32 32, label %clear_pmop
		 i32 33, label %bb101
	]

bb101:		; preds = %entry
	%tmp102103 = bitcast %struct.OP* %o to %struct.PMOP*		; <%struct.PMOP*> [#uses=1]
	%tmp104 = getelementptr %struct.PMOP* %tmp102103, i32 0, i32 10		; <%struct.OP**> [#uses=0]
	br i1 false, label %cond_next174, label %cond_true122

bb106:		; preds = %entry, %entry
	%tmp107108 = bitcast %struct.OP* %o to %struct.PMOP*		; <%struct.PMOP*> [#uses=0]
	br label %clear_pmop

clear_pmop:		; preds = %bb106, %entry, %entry
	%tmp114115 = bitcast %struct.OP* %o to %struct.PMOP*		; <%struct.PMOP*> [#uses=0]
	br label %cond_true122

cond_true122:		; preds = %clear_pmop, %bb101
	br i1 false, label %cond_next174, label %cond_true129

cond_true129:		; preds = %cond_true122
	ret void

cond_next174:		; preds = %cond_true122, %bb101
	%tmp175176 = bitcast %struct.OP* %o to %struct.PMOP*		; <%struct.PMOP*> [#uses=1]
	%tmp177 = getelementptr %struct.PMOP* %tmp175176, i32 0, i32 10		; <%struct.OP**> [#uses=0]
	ret void
}
