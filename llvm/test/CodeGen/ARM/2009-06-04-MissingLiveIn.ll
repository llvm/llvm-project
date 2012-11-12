; RUN: llc < %s -mtriple=arm-apple-darwin -mattr=+v6

	%struct.anon = type { i16, i16 }
	%struct.cab_archive = type { i32, i16, i16, i16, i16, i8, %struct.cab_folder*, %struct.cab_file* }
	%struct.cab_file = type { i32, i16, i64, i8*, i32, i32, i32, %struct.cab_folder*, %struct.cab_file*, %struct.cab_archive*, %struct.cab_state* }
	%struct.cab_folder = type { i16, i16, %struct.cab_archive*, i64, %struct.cab_folder* }
	%struct.cab_state = type { i8*, i8*, [38912 x i8], i16, i16, i8*, i16 }
	%struct.qtm_model = type { i32, i32, %struct.anon* }
	%struct.qtm_stream = type { i32, i32, i8, i8*, i32, i32, i32, i16, i16, i16, i8, i32, i8*, i8*, i8*, i8*, i8*, i32, i32, i8, [42 x i32], [42 x i8], [27 x i8], [27 x i8], %struct.qtm_model, %struct.qtm_model, %struct.qtm_model, %struct.qtm_model, %struct.qtm_model, %struct.qtm_model, %struct.qtm_model, %struct.qtm_model, %struct.qtm_model, [65 x %struct.anon], [65 x %struct.anon], [65 x %struct.anon], [65 x %struct.anon], [25 x %struct.anon], [37 x %struct.anon], [43 x %struct.anon], [28 x %struct.anon], [8 x %struct.anon], %struct.cab_file*, i32 (%struct.cab_file*, i8*, i32)* }

declare fastcc i32 @qtm_read_input(%struct.qtm_stream* nocapture) nounwind

define fastcc i32 @qtm_decompress(%struct.qtm_stream* %qtm, i64 %out_bytes) nounwind {
entry:
	br i1 undef, label %bb245, label %bb3

bb3:		; preds = %entry
	br i1 undef, label %bb5, label %bb4

bb4:		; preds = %bb3
	ret i32 undef

bb5:		; preds = %bb3
	br i1 undef, label %bb245, label %bb14

bb14:		; preds = %bb5
	br label %bb238

bb28:		; preds = %bb215
	br label %bb31

bb29:		; preds = %bb31
	br i1 undef, label %bb31, label %bb32

bb31:		; preds = %bb29, %bb28
	br i1 undef, label %bb29, label %bb32

bb32:		; preds = %bb31, %bb29
	br label %bb33

bb33:		; preds = %bb33, %bb32
	br i1 undef, label %bb34, label %bb33

bb34:		; preds = %bb33
	br i1 undef, label %bb35, label %bb36

bb35:		; preds = %bb34
	br label %bb36

bb36:		; preds = %bb46, %bb35, %bb34
	br i1 undef, label %bb40, label %bb37

bb37:		; preds = %bb36
	br i1 undef, label %bb77, label %bb60

bb40:		; preds = %bb36
	br i1 undef, label %bb46, label %bb41

bb41:		; preds = %bb40
	br i1 undef, label %bb45, label %bb42

bb42:		; preds = %bb41
	ret i32 undef

bb45:		; preds = %bb41
	br label %bb46

bb46:		; preds = %bb45, %bb40
	br label %bb36

bb60:		; preds = %bb60, %bb37
	br label %bb60

bb77:		; preds = %bb37
	switch i32 undef, label %bb197 [
		i32 5, label %bb108
		i32 6, label %bb138
	]

bb108:		; preds = %bb77
	br label %bb111

bb109:		; preds = %bb111
	br i1 undef, label %bb111, label %bb112

bb111:		; preds = %bb109, %bb108
	br i1 undef, label %bb109, label %bb112

bb112:		; preds = %bb111, %bb109
	br label %bb113

bb113:		; preds = %bb113, %bb112
	br i1 undef, label %bb114, label %bb113

bb114:		; preds = %bb113
	br i1 undef, label %bb115, label %bb116

bb115:		; preds = %bb114
	br label %bb116

bb116:		; preds = %bb115, %bb114
	br i1 undef, label %bb120, label %bb117

bb117:		; preds = %bb116
	br label %bb136

bb120:		; preds = %bb116
	ret i32 undef

bb128:		; preds = %bb136
	br i1 undef, label %bb134, label %bb129

bb129:		; preds = %bb128
	br i1 undef, label %bb133, label %bb130

bb130:		; preds = %bb129
	br i1 undef, label %bb132, label %bb131

bb131:		; preds = %bb130
	ret i32 undef

bb132:		; preds = %bb130
	br label %bb133

bb133:		; preds = %bb132, %bb129
	br label %bb134

bb134:		; preds = %bb133, %bb128
	br label %bb136

bb136:		; preds = %bb134, %bb117
	br i1 undef, label %bb198, label %bb128

bb138:		; preds = %bb77
	%0 = trunc i32 undef to i16		; <i16> [#uses=1]
	br label %bb141

bb139:		; preds = %bb141
	%scevgep441442881 = load i16* undef		; <i16> [#uses=1]
	%1 = icmp ugt i16 %scevgep441442881, %0		; <i1> [#uses=1]
	br i1 %1, label %bb141, label %bb142

bb141:		; preds = %bb139, %bb138
	br i1 undef, label %bb139, label %bb142

bb142:		; preds = %bb141, %bb139
	br label %bb143

bb143:		; preds = %bb143, %bb142
	br i1 undef, label %bb144, label %bb143

bb144:		; preds = %bb143
	br i1 undef, label %bb145, label %bb146

bb145:		; preds = %bb144
	unreachable

bb146:		; preds = %bb156, %bb144
	br i1 undef, label %bb150, label %bb147

bb147:		; preds = %bb146
	br i1 undef, label %bb157, label %bb148

bb148:		; preds = %bb147
	br i1 undef, label %bb149, label %bb157

bb149:		; preds = %bb148
	br label %bb150

bb150:		; preds = %bb149, %bb146
	br i1 undef, label %bb156, label %bb152

bb152:		; preds = %bb150
	unreachable

bb156:		; preds = %bb150
	br label %bb146

bb157:		; preds = %bb148, %bb147
	br i1 undef, label %bb167, label %bb160

bb160:		; preds = %bb157
	ret i32 undef

bb167:		; preds = %bb157
	br label %bb170

bb168:		; preds = %bb170
	br i1 undef, label %bb170, label %bb171

bb170:		; preds = %bb168, %bb167
	br i1 undef, label %bb168, label %bb171

bb171:		; preds = %bb170, %bb168
	br label %bb172

bb172:		; preds = %bb172, %bb171
	br i1 undef, label %bb173, label %bb172

bb173:		; preds = %bb172
	br i1 undef, label %bb174, label %bb175

bb174:		; preds = %bb173
	unreachable

bb175:		; preds = %bb179, %bb173
	br i1 undef, label %bb179, label %bb176

bb176:		; preds = %bb175
	br i1 undef, label %bb186, label %bb177

bb177:		; preds = %bb176
	br i1 undef, label %bb178, label %bb186

bb178:		; preds = %bb177
	br label %bb179

bb179:		; preds = %bb178, %bb175
	br label %bb175

bb186:		; preds = %bb177, %bb176
	br label %bb195

bb187:		; preds = %bb195
	br i1 undef, label %bb193, label %bb189

bb189:		; preds = %bb187
	%2 = tail call fastcc i32 @qtm_read_input(%struct.qtm_stream* %qtm) nounwind		; <i32> [#uses=0]
	ret i32 undef

bb193:		; preds = %bb187
	br label %bb195

bb195:		; preds = %bb193, %bb186
	br i1 undef, label %bb198, label %bb187

bb197:		; preds = %bb77
	ret i32 -124

bb198:		; preds = %bb195, %bb136
	br i1 undef, label %bb211.preheader, label %bb214

bb211.preheader:		; preds = %bb198
	br label %bb211

bb211:		; preds = %bb211, %bb211.preheader
	br i1 undef, label %bb214, label %bb211

bb214:		; preds = %bb211, %bb198
	br label %bb215

bb215:		; preds = %bb238, %bb214
	br i1 undef, label %bb28, label %bb216

bb216:		; preds = %bb215
	br label %bb238

bb238:		; preds = %bb216, %bb14
	br label %bb215

bb245:		; preds = %bb5, %entry
	ret i32 undef
}
