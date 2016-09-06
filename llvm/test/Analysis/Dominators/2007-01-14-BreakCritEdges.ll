; RUN: opt < %s -domtree -break-crit-edges -disable-output
; RUN: opt < %s -passes='require<domtree>,break-crit-edges' -disable-output
; PR1110

	%struct.OggVorbis_File = type { i8*, i32, i64, i64, %struct.ogg_sync_state, i32, i64*, i64*, i32*, i64*, %struct.vorbis_info*, %struct.vorbis_comment*, i64, i32, i32, i32, double, double, %struct.ogg_stream_state, %struct.vorbis_dsp_state, %struct.vorbis_block, %struct.ov_callbacks }
	%struct.alloc_chain = type { i8*, %struct.alloc_chain* }
	%struct.ogg_stream_state = type { i8*, i32, i32, i32, i32*, i64*, i32, i32, i32, i32, [282 x i8], i32, i32, i32, i32, i32, i64, i64 }
	%struct.ogg_sync_state = type { i8*, i32, i32, i32, i32, i32, i32 }
	%struct.oggpack_buffer = type { i32, i32, i8*, i8*, i32 }
	%struct.ov_callbacks = type { i32 (i8*, i32, i32, i8*)*, i32 (i8*, i64, i32)*, i32 (i8*)*, i32 (i8*)* }
	%struct.vorbis_block = type { float**, %struct.oggpack_buffer, i32, i32, i32, i32, i32, i32, i64, i64, %struct.vorbis_dsp_state*, i8*, i32, i32, i32, %struct.alloc_chain*, i32, i32, i32, i32, i8* }
	%struct.vorbis_comment = type { i8**, i32*, i32, i8* }
	%struct.vorbis_dsp_state = type { i32, %struct.vorbis_info*, float**, float**, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i8* }
	%struct.vorbis_info = type { i32, i32, i32, i32, i32, i32, i32, i8* }


define void @ov_read() {
entry:
	br i1 false, label %bb, label %return

bb:		; preds = %cond_next22, %entry
	br i1 false, label %cond_true8, label %cond_next15

cond_true8:		; preds = %bb
	br i1 false, label %cond_next15, label %bb29

cond_next15:		; preds = %cond_true8, %bb
	br i1 false, label %return, label %cond_next22

cond_next22:		; preds = %cond_next15
	br i1 false, label %bb, label %return

bb29:		; preds = %cond_true8
	br i1 false, label %cond_true32, label %return

cond_true32:		; preds = %bb29
	br i1 false, label %cond_false37.i, label %cond_true.i11

cond_true.i11:		; preds = %cond_true32
	br i1 false, label %cond_true8.i, label %ov_info.exit

cond_true8.i:		; preds = %cond_true.i11
	br i1 false, label %cond_true44, label %cond_next48

cond_false37.i:		; preds = %cond_true32
	br label %ov_info.exit

ov_info.exit:		; preds = %cond_false37.i, %cond_true.i11
	br i1 false, label %cond_true44, label %cond_next48

cond_true44:		; preds = %ov_info.exit, %cond_true8.i
	br label %cond_next48

cond_next48:		; preds = %cond_true44, %ov_info.exit, %cond_true8.i
	br i1 false, label %cond_next53, label %return

cond_next53:		; preds = %cond_next48
	br i1 false, label %cond_true56, label %cond_false97

cond_true56:		; preds = %cond_next53
	br i1 false, label %bb85, label %cond_next304

bb63:		; preds = %bb85
	br i1 false, label %cond_next78, label %cond_false73

cond_false73:		; preds = %bb63
	br i1 false, label %cond_true76, label %cond_next78

cond_true76:		; preds = %cond_false73
	br label %cond_next78

cond_next78:		; preds = %cond_true76, %cond_false73, %bb63
	br label %bb85

bb85:		; preds = %bb89, %cond_next78, %cond_true56
	br i1 false, label %bb63, label %bb89

bb89:		; preds = %bb85
	br i1 false, label %bb85, label %cond_next304

cond_false97:		; preds = %cond_next53
	br i1 false, label %cond_true108, label %bb248

cond_true108:		; preds = %cond_false97
	br i1 false, label %bb196, label %bb149

bb112:		; preds = %bb149, %bb146
	br i1 false, label %bb119, label %bb146

bb119:		; preds = %cond_next134, %bb112
	br i1 false, label %cond_next134, label %cond_false129

cond_false129:		; preds = %bb119
	br i1 false, label %cond_true132, label %cond_next134

cond_true132:		; preds = %cond_false129
	br label %cond_next134

cond_next134:		; preds = %cond_true132, %cond_false129, %bb119
	br i1 false, label %bb119, label %bb146

bb146:		; preds = %cond_next134, %bb112
	br i1 false, label %bb112, label %cond_next304

bb149:		; preds = %cond_true108
	br i1 false, label %bb112, label %cond_next304

bb155:		; preds = %bb196, %bb193
	br i1 false, label %bb165, label %bb193

bb165:		; preds = %cond_next180, %bb155
	br i1 false, label %cond_next180, label %cond_false175

cond_false175:		; preds = %bb165
	br i1 false, label %cond_true178, label %cond_next180

cond_true178:		; preds = %cond_false175
	br label %cond_next180

cond_next180:		; preds = %cond_true178, %cond_false175, %bb165
	br i1 false, label %bb165, label %bb193

bb193:		; preds = %cond_next180, %bb155
	br i1 false, label %bb155, label %cond_next304

bb196:		; preds = %cond_true108
	br i1 false, label %bb155, label %cond_next304

bb207:		; preds = %bb241
	br i1 false, label %cond_next225, label %cond_false220

cond_false220:		; preds = %bb207
	br i1 false, label %cond_true223, label %cond_next225

cond_true223:		; preds = %cond_false220
	br label %cond_next225

cond_next225:		; preds = %cond_true223, %cond_false220, %bb207
	br label %bb241

bb241:		; preds = %bb248, %bb245, %cond_next225
	br i1 false, label %bb207, label %bb245

bb245:		; preds = %bb241
	br i1 false, label %bb241, label %cond_next304

bb248:		; preds = %cond_false97
	br i1 false, label %bb241, label %cond_next304

bb256:		; preds = %bb290
	br i1 false, label %cond_next274, label %cond_false269

cond_false269:		; preds = %bb256
	br i1 false, label %cond_true272, label %cond_next274

cond_true272:		; preds = %cond_false269
	br label %cond_next274

cond_next274:		; preds = %cond_true272, %cond_false269, %bb256
	br label %bb290

bb290:		; preds = %bb294, %cond_next274
	br i1 false, label %bb256, label %bb294

bb294:		; preds = %bb290
	br i1 false, label %bb290, label %cond_next304

cond_next304:		; preds = %bb294, %bb248, %bb245, %bb196, %bb193, %bb149, %bb146, %bb89, %cond_true56
	br i1 false, label %cond_next11.i, label %cond_true.i

cond_true.i:		; preds = %cond_next304
	br i1 false, label %vorbis_synthesis_read.exit, label %cond_next11.i

cond_next11.i:		; preds = %cond_true.i, %cond_next304
	br label %vorbis_synthesis_read.exit

vorbis_synthesis_read.exit:		; preds = %cond_next11.i, %cond_true.i
	br i1 false, label %cond_next321, label %cond_true316

cond_true316:		; preds = %vorbis_synthesis_read.exit
	ret void

cond_next321:		; preds = %vorbis_synthesis_read.exit
	ret void

return:		; preds = %cond_next48, %bb29, %cond_next22, %cond_next15, %entry
	ret void
}
