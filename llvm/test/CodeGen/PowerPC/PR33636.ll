; Just a test case for a crash reported in
; https://bugs.llvm.org/show_bug.cgi?id=33636
; RUN: llc -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 < %s | FileCheck %s
@g_225 = external unnamed_addr global i16, align 2
@g_756 = external global [6 x i32], align 4
@g_3456 = external global i32, align 4
@g_3708 = external global [9 x i32], align 4
@g_1252 = external global i8*, align 8
@g_3043 = external global float*, align 8

; Function Attrs: nounwind
define void @main() {
  br i1 undef, label %1, label %4

; <label>:1:                                      ; preds = %0
  br i1 undef, label %2, label %3

; <label>:2:                                      ; preds = %1
  br label %3

; <label>:3:                                      ; preds = %2, %1
  br label %4

; <label>:4:                                      ; preds = %3, %0
  br label %5

; <label>:5:                                      ; preds = %5, %4
  br i1 undef, label %6, label %5

; <label>:6:                                      ; preds = %5
  br i1 undef, label %7, label %8

; <label>:7:                                      ; preds = %6
  br i1 undef, label %70, label %69

; <label>:8:                                      ; preds = %6
  br i1 undef, label %9, label %50

; <label>:9:                                      ; preds = %8
  br label %11

; <label>:10:                                     ; preds = %28
  br i1 undef, label %11, label %12

; <label>:11:                                     ; preds = %10, %9
  br label %13

; <label>:12:                                     ; preds = %10
  br label %30

; <label>:13:                                     ; preds = %23, %11
  br i1 undef, label %17, label %14

; <label>:14:                                     ; preds = %13
  br i1 undef, label %16, label %15

; <label>:15:                                     ; preds = %14
  br label %22

; <label>:16:                                     ; preds = %14
  br label %17

; <label>:17:                                     ; preds = %16, %13
  br i1 undef, label %18, label %19

; <label>:18:                                     ; preds = %17
  br label %19

; <label>:19:                                     ; preds = %18, %17
  br i1 undef, label %48, label %20

; <label>:20:                                     ; preds = %19
  br i1 undef, label %48, label %21

; <label>:21:                                     ; preds = %20
  br label %22

; <label>:22:                                     ; preds = %21, %15
  br i1 undef, label %23, label %24

; <label>:23:                                     ; preds = %22
  br label %13

; <label>:24:                                     ; preds = %22
  br i1 undef, label %28, label %25

; <label>:25:                                     ; preds = %24
  br label %26

; <label>:26:                                     ; preds = %26, %25
  br i1 undef, label %26, label %27

; <label>:27:                                     ; preds = %26
  br label %48

; <label>:28:                                     ; preds = %24
  br i1 undef, label %29, label %10

; <label>:29:                                     ; preds = %28
  br label %48

; <label>:30:                                     ; preds = %33, %12
  br i1 undef, label %32, label %33

; <label>:31:                                     ; preds = %33
  br label %34

; <label>:32:                                     ; preds = %30
  br label %33

; <label>:33:                                     ; preds = %32, %30
  br i1 undef, label %30, label %31

; <label>:34:                                     ; preds = %47, %31
  br i1 undef, label %35, label %36

; <label>:35:                                     ; preds = %34
  br label %36

; <label>:36:                                     ; preds = %35, %34
  br label %37

; <label>:37:                                     ; preds = %45, %36
  br i1 undef, label %40, label %38

; <label>:38:                                     ; preds = %37
  br i1 undef, label %39, label %46

; <label>:39:                                     ; preds = %38
  br label %41

; <label>:40:                                     ; preds = %37
  br label %41

; <label>:41:                                     ; preds = %40, %39
  br label %42

; <label>:42:                                     ; preds = %44, %41
  br i1 undef, label %43, label %44

; <label>:43:                                     ; preds = %42
  br label %44

; <label>:44:                                     ; preds = %43, %42
  br i1 undef, label %42, label %45

; <label>:45:                                     ; preds = %44
  br i1 undef, label %37, label %47

; <label>:46:                                     ; preds = %38
  br label %48

; <label>:47:                                     ; preds = %45
  br i1 undef, label %34, label %49

; <label>:48:                                     ; preds = %46, %29, %27, %20, %19
  br label %65

; <label>:49:                                     ; preds = %47
  br label %58

; <label>:50:                                     ; preds = %8
  br i1 undef, label %52, label %51

; <label>:51:                                     ; preds = %50
  br label %57

; <label>:52:                                     ; preds = %50
  br label %53

; <label>:53:                                     ; preds = %56, %52
  br i1 undef, label %54, label %59

; <label>:54:                                     ; preds = %53
  br i1 undef, label %60, label %59

; <label>:55:                                     ; preds = %64
  br label %56

; <label>:56:                                     ; preds = %64, %55
  br i1 undef, label %57, label %53

; <label>:57:                                     ; preds = %56, %51
  br label %58

; <label>:58:                                     ; preds = %57, %49
  br label %65

; <label>:59:                                     ; preds = %63, %62, %61, %60, %54, %53
  br label %65

; <label>:60:                                     ; preds = %54
  br i1 undef, label %61, label %59

; <label>:61:                                     ; preds = %60
  br i1 undef, label %62, label %59

; <label>:62:                                     ; preds = %61
  br i1 undef, label %63, label %59

; <label>:63:                                     ; preds = %62
  br i1 undef, label %64, label %59

; <label>:64:                                     ; preds = %63
  br i1 undef, label %55, label %56

; <label>:65:                                     ; preds = %59, %58, %48
  br i1 undef, label %66, label %67

; <label>:66:                                     ; preds = %65
  br label %67

; <label>:67:                                     ; preds = %66, %65
  br i1 undef, label %68, label %92

; <label>:68:                                     ; preds = %67
  br label %92

; <label>:69:                                     ; preds = %7
  br label %70

; <label>:70:                                     ; preds = %69, %7
  br i1 undef, label %72, label %71

; <label>:71:                                     ; preds = %70
  br label %72

; <label>:72:                                     ; preds = %71, %70
  br i1 undef, label %73, label %74

; <label>:73:                                     ; preds = %72
  br label %74

; <label>:74:                                     ; preds = %73, %72
  br i1 undef, label %85, label %75

; <label>:75:                                     ; preds = %74
  br i1 undef, label %84, label %76

; <label>:76:                                     ; preds = %75
  br i1 undef, label %78, label %77

; <label>:77:                                     ; preds = %77, %76
  br i1 undef, label %84, label %77

; <label>:78:                                     ; preds = %76
  br label %79

; <label>:79:                                     ; preds = %83, %78
  br i1 undef, label %83, label %80

; <label>:80:                                     ; preds = %79
  br i1 undef, label %81, label %82

; <label>:81:                                     ; preds = %80
  br label %83

; <label>:82:                                     ; preds = %80
  br label %83

; <label>:83:                                     ; preds = %82, %81, %79
  br i1 undef, label %90, label %79

; <label>:84:                                     ; preds = %77, %75
  br label %92

; <label>:85:                                     ; preds = %74
  br i1 undef, label %86, label %88

; <label>:86:                                     ; preds = %85
  br i1 undef, label %89, label %87

; <label>:87:                                     ; preds = %86
  br i1 undef, label %89, label %88

; <label>:88:                                     ; preds = %87, %85
  br label %89

; <label>:89:                                     ; preds = %88, %87, %86
  br label %92

; <label>:90:                                     ; preds = %83
  br i1 undef, label %92, label %91

; <label>:91:                                     ; preds = %90
  br label %92

; <label>:92:                                     ; preds = %91, %90, %89, %84, %68, %67
  br label %93

; <label>:93:                                     ; preds = %100, %92
  br label %94

; <label>:94:                                     ; preds = %98, %93
  br label %95

; <label>:95:                                     ; preds = %97, %94
  br i1 undef, label %96, label %97

; <label>:96:                                     ; preds = %95
  br label %97

; <label>:97:                                     ; preds = %96, %95
  br i1 undef, label %95, label %98

; <label>:98:                                     ; preds = %97
  store i32 7, i32* getelementptr inbounds ([9 x i32], [9 x i32]* @g_3708, i64 0, i64 7), align 4
  %99 = load volatile i32, i32* @g_3456, align 4
  br i1 undef, label %94, label %100

; <label>:100:                                    ; preds = %98
  br i1 undef, label %93, label %101

; <label>:101:                                    ; preds = %100
  br label %102

; <label>:102:                                    ; preds = %117, %101
  br label %103

; <label>:103:                                    ; preds = %109, %102
  store i8** @g_1252, i8*** undef, align 8
  br i1 undef, label %105, label %104

; <label>:104:                                    ; preds = %103
  br label %105

; <label>:105:                                    ; preds = %104, %103
  %106 = icmp eq i32 0, 0
  br i1 %106, label %107, label %116

; <label>:107:                                    ; preds = %105
  br i1 icmp ne (i32* getelementptr inbounds ([6 x i32], [6 x i32]* @g_756, i64 0, i64 0), i32* getelementptr inbounds ([9 x i32], [9 x i32]* @g_3708, i64 0, i64 4)), label %109, label %108

; <label>:108:                                    ; preds = %107
  br label %109

; <label>:109:                                    ; preds = %108, %107
  %110 = phi i32 [ sdiv (i32 32, i32 zext (i1 icmp eq (i32* getelementptr inbounds ([6 x i32], [6 x i32]* @g_756, i64 0, i64 0), i32* getelementptr inbounds ([9 x i32], [9 x i32]* @g_3708, i64 0, i64 4)) to i32)), %108 ], [ 32, %107 ]
  %111 = trunc i32 %110 to i8
  %112 = icmp ne i8 %111, 0
  %113 = and i1 %112, icmp eq (i32* getelementptr inbounds ([6 x i32], [6 x i32]* @g_756, i64 0, i64 0), i32* getelementptr inbounds ([9 x i32], [9 x i32]* @g_3708, i64 0, i64 4))
  %114 = zext i1 %113 to i16
  store i16 %114, i16* @g_225, align 2
  %115 = load volatile float*, float** @g_3043, align 8
  br i1 undef, label %103, label %117

; <label>:116:                                    ; preds = %105
  br label %119

; <label>:117:                                    ; preds = %109
  br i1 undef, label %102, label %118

; <label>:118:                                    ; preds = %117
  br label %119

; <label>:119:                                    ; preds = %118, %116
  br i1 undef, label %120, label %231

; <label>:120:                                    ; preds = %119
  br label %232

; <label>:121:                                    ; preds = %230
  br label %122

; <label>:122:                                    ; preds = %230, %121
  br i1 undef, label %124, label %123

; <label>:123:                                    ; preds = %122
  br label %124

; <label>:124:                                    ; preds = %123, %122
  br i1 undef, label %228, label %225

; <label>:125:                                    ; preds = %218
  br label %127

; <label>:126:                                    ; preds = %218
  br label %127

; <label>:127:                                    ; preds = %216, %126, %125
  br i1 undef, label %204, label %128

; <label>:128:                                    ; preds = %127
  br label %205

; <label>:129:                                    ; preds = %216
  br i1 undef, label %131, label %130

; <label>:130:                                    ; preds = %129
  br label %131

; <label>:131:                                    ; preds = %130, %129
  br i1 undef, label %133, label %132

; <label>:132:                                    ; preds = %131
  br label %133

; <label>:133:                                    ; preds = %132, %131
  br label %134

; <label>:134:                                    ; preds = %203, %133
  br i1 undef, label %193, label %135

; <label>:135:                                    ; preds = %134
  br label %194

; <label>:136:                                    ; preds = %203
  br i1 undef, label %138, label %137

; <label>:137:                                    ; preds = %136
  br label %138

; <label>:138:                                    ; preds = %137, %136
  br i1 undef, label %192, label %139

; <label>:139:                                    ; preds = %138
  br label %191

; <label>:140:                                    ; preds = %191, %190
  br i1 undef, label %180, label %141

; <label>:141:                                    ; preds = %140
  br label %181

; <label>:142:                                    ; preds = %190
  br i1 undef, label %143, label %178

; <label>:143:                                    ; preds = %142
  br label %179

; <label>:144:                                    ; preds = %179
  br label %176

; <label>:145:                                    ; preds = %179
  br label %176

; <label>:146:                                    ; preds = %177, %175, %174
  br i1 undef, label %165, label %147

; <label>:147:                                    ; preds = %146
  br label %166

; <label>:148:                                    ; preds = %174
  br label %149

; <label>:149:                                    ; preds = %164, %148
  br i1 undef, label %154, label %150

; <label>:150:                                    ; preds = %149
  br label %155

; <label>:151:                                    ; preds = %164
  br i1 undef, label %153, label %152

; <label>:152:                                    ; preds = %151
  br label %153

; <label>:153:                                    ; preds = %152, %151
  ret void

; <label>:154:                                    ; preds = %149
  br label %155

; <label>:155:                                    ; preds = %154, %150
  br i1 undef, label %157, label %156

; <label>:156:                                    ; preds = %155
  br label %158

; <label>:157:                                    ; preds = %155
  br label %158

; <label>:158:                                    ; preds = %157, %156
  br i1 undef, label %160, label %159

; <label>:159:                                    ; preds = %158
  br label %161

; <label>:160:                                    ; preds = %158
  br label %161

; <label>:161:                                    ; preds = %160, %159
  br i1 undef, label %163, label %162

; <label>:162:                                    ; preds = %161
  br label %164

; <label>:163:                                    ; preds = %161
  br label %164

; <label>:164:                                    ; preds = %163, %162
  br i1 undef, label %151, label %149

; <label>:165:                                    ; preds = %146
  br label %166

; <label>:166:                                    ; preds = %165, %147
  br i1 undef, label %168, label %167

; <label>:167:                                    ; preds = %166
  br label %169

; <label>:168:                                    ; preds = %166
  br label %169

; <label>:169:                                    ; preds = %168, %167
  br i1 undef, label %171, label %170

; <label>:170:                                    ; preds = %169
  br label %172

; <label>:171:                                    ; preds = %169
  br label %172

; <label>:172:                                    ; preds = %171, %170
  br i1 undef, label %174, label %173

; <label>:173:                                    ; preds = %172
  br label %174

; <label>:174:                                    ; preds = %173, %172
  br i1 undef, label %148, label %146

; <label>:175:                                    ; preds = %176
  br label %146

; <label>:176:                                    ; preds = %145, %144
  br i1 undef, label %177, label %175

; <label>:177:                                    ; preds = %176
  br label %146

; <label>:178:                                    ; preds = %142
  br label %179

; <label>:179:                                    ; preds = %178, %143
  br i1 undef, label %145, label %144

; <label>:180:                                    ; preds = %140
  br label %181

; <label>:181:                                    ; preds = %180, %141
  br i1 undef, label %183, label %182

; <label>:182:                                    ; preds = %181
  br label %184

; <label>:183:                                    ; preds = %181
  br label %184

; <label>:184:                                    ; preds = %183, %182
  br i1 undef, label %186, label %185

; <label>:185:                                    ; preds = %184
  br label %187

; <label>:186:                                    ; preds = %184
  br label %187

; <label>:187:                                    ; preds = %186, %185
  br i1 undef, label %189, label %188

; <label>:188:                                    ; preds = %187
  br label %190

; <label>:189:                                    ; preds = %187
  br label %190

; <label>:190:                                    ; preds = %189, %188
  br i1 undef, label %142, label %140

; <label>:191:                                    ; preds = %192, %139
  br label %140

; <label>:192:                                    ; preds = %138
  br label %191

; <label>:193:                                    ; preds = %134
  br label %194

; <label>:194:                                    ; preds = %193, %135
  br i1 undef, label %196, label %195

; <label>:195:                                    ; preds = %194
  br label %197

; <label>:196:                                    ; preds = %194
  br label %197

; <label>:197:                                    ; preds = %196, %195
  br i1 undef, label %199, label %198

; <label>:198:                                    ; preds = %197
  br label %200

; <label>:199:                                    ; preds = %197
  br label %200

; <label>:200:                                    ; preds = %199, %198
  br i1 undef, label %202, label %201

; <label>:201:                                    ; preds = %200
  br label %203

; <label>:202:                                    ; preds = %200
  br label %203

; <label>:203:                                    ; preds = %202, %201
  br i1 undef, label %136, label %134

; <label>:204:                                    ; preds = %127
  br label %205

; <label>:205:                                    ; preds = %204, %128
  br i1 undef, label %207, label %206

; <label>:206:                                    ; preds = %205
  br label %208

; <label>:207:                                    ; preds = %205
  br label %208

; <label>:208:                                    ; preds = %207, %206
  br i1 undef, label %210, label %209

; <label>:209:                                    ; preds = %208
  br label %211

; <label>:210:                                    ; preds = %208
  br label %211

; <label>:211:                                    ; preds = %210, %209
  br i1 undef, label %213, label %212

; <label>:212:                                    ; preds = %211
  br label %214

; <label>:213:                                    ; preds = %211
  br label %214

; <label>:214:                                    ; preds = %213, %212
  br i1 undef, label %216, label %215

; <label>:215:                                    ; preds = %214
  br label %216

; <label>:216:                                    ; preds = %215, %214
  br i1 undef, label %129, label %127

; <label>:217:                                    ; preds = %220
  br label %218

; <label>:218:                                    ; preds = %221, %217
  br i1 undef, label %126, label %125

; <label>:219:                                    ; preds = %223
  br label %220

; <label>:220:                                    ; preds = %224, %219
  br i1 undef, label %221, label %217

; <label>:221:                                    ; preds = %220
  br label %218

; <label>:222:                                    ; preds = %226
  br label %223

; <label>:223:                                    ; preds = %227, %222
  br i1 undef, label %224, label %219

; <label>:224:                                    ; preds = %223
  br label %220

; <label>:225:                                    ; preds = %124
  br label %226

; <label>:226:                                    ; preds = %228, %225
  br i1 undef, label %227, label %222

; <label>:227:                                    ; preds = %226
  br label %223

; <label>:228:                                    ; preds = %124
  br label %226

; <label>:229:                                    ; preds = %232
  br label %230

; <label>:230:                                    ; preds = %233, %229
  br i1 undef, label %122, label %121

; <label>:231:                                    ; preds = %119
  br label %232

; <label>:232:                                    ; preds = %231, %120
  br i1 undef, label %233, label %229

; <label>:233:                                    ; preds = %232
  br label %230

; CHECK: blr
}
