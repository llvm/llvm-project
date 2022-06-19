; RUN: opt < %s -passes=newgvn | llvm-dis

	%struct.PerlInterpreter = type { i8 }
@PL_sv_count = external global i32		; <i32*> [#uses=2]

define void @perl_destruct(%struct.PerlInterpreter* %sv_interp) {
entry:
	br i1 false, label %cond_next25, label %cond_true16

cond_true16:		; preds = %entry
	ret void

cond_next25:		; preds = %entry
	br i1 false, label %cond_next33, label %cond_true32

cond_true32:		; preds = %cond_next25
	ret void

cond_next33:		; preds = %cond_next25
	br i1 false, label %cond_next61, label %cond_true.i46

cond_true.i46:		; preds = %cond_next33
	ret void

cond_next61:		; preds = %cond_next33
	br i1 false, label %cond_next69, label %cond_true66

cond_true66:		; preds = %cond_next61
	ret void

cond_next69:		; preds = %cond_next61
	br i1 false, label %Perl_safefree.exit52, label %cond_true.i50

cond_true.i50:		; preds = %cond_next69
	ret void

Perl_safefree.exit52:		; preds = %cond_next69
	br i1 false, label %cond_next80, label %cond_true77

cond_true77:		; preds = %Perl_safefree.exit52
	ret void

cond_next80:		; preds = %Perl_safefree.exit52
	br i1 false, label %Perl_safefree.exit56, label %cond_true.i54

cond_true.i54:		; preds = %cond_next80
	ret void

Perl_safefree.exit56:		; preds = %cond_next80
	br i1 false, label %Perl_safefree.exit60, label %cond_true.i58

cond_true.i58:		; preds = %Perl_safefree.exit56
	ret void

Perl_safefree.exit60:		; preds = %Perl_safefree.exit56
	br i1 false, label %Perl_safefree.exit64, label %cond_true.i62

cond_true.i62:		; preds = %Perl_safefree.exit60
	ret void

Perl_safefree.exit64:		; preds = %Perl_safefree.exit60
	br i1 false, label %Perl_safefree.exit68, label %cond_true.i66

cond_true.i66:		; preds = %Perl_safefree.exit64
	ret void

Perl_safefree.exit68:		; preds = %Perl_safefree.exit64
	br i1 false, label %cond_next150, label %cond_true23.i

cond_true23.i:		; preds = %Perl_safefree.exit68
	ret void

cond_next150:		; preds = %Perl_safefree.exit68
	%tmp16092 = load i32, i32* @PL_sv_count, align 4		; <i32> [#uses=0]
	br label %cond_next165

bb157:		; preds = %cond_next165
	%tmp158 = load i32, i32* @PL_sv_count, align 4		; <i32> [#uses=0]
	br label %cond_next165

cond_next165:		; preds = %bb157, %cond_next150
	br i1 false, label %bb171, label %bb157

bb171:		; preds = %cond_next165
	ret void
}
