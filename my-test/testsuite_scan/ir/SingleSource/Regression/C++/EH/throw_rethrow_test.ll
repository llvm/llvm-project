; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/throw_rethrow_test.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/throw_rethrow_test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

$__clang_call_terminate = comdat any

$_ZTI3foo = comdat any

$_ZTS3foo = comdat any

@_ZTIi = external constant ptr
@_ZTId = external constant ptr
@_ZTI3foo = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS3foo }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS3foo = linkonce_odr dso_local constant [5 x i8] c"3foo\00", comdat, align 1
@.str = private unnamed_addr constant [7 x i8] c"%d: 2\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"%d: 1\0A\00", align 1
@.str.2 = private unnamed_addr constant [7 x i8] c"%d: 3\0A\00", align 1

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z6calleej(i32 noundef %0) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %2 = icmp ult i32 %0, 3
  br i1 %2, label %3, label %5

3:                                                ; preds = %1
  %4 = tail call ptr @__cxa_allocate_exception(i64 4) #8
  store i32 %0, ptr %4, align 16, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTIi, ptr null) #9
  unreachable

5:                                                ; preds = %1
  %6 = icmp ult i32 %0, 6
  br i1 %6, label %7, label %9

7:                                                ; preds = %5
  %8 = tail call ptr @__cxa_allocate_exception(i64 8) #8
  store double 1.000000e+00, ptr %8, align 16, !tbaa !10
  tail call void @__cxa_throw(ptr nonnull %8, ptr nonnull @_ZTId, ptr null) #9
  unreachable

9:                                                ; preds = %5
  %10 = icmp ult i32 %0, 9
  br i1 %10, label %11, label %13

11:                                               ; preds = %9
  %12 = tail call ptr @__cxa_allocate_exception(i64 4) #8
  store i32 1, ptr %12, align 4, !tbaa !12
  tail call void @__cxa_throw(ptr nonnull %12, ptr nonnull @_ZTI3foo, ptr null) #9
  unreachable

13:                                               ; preds = %9
  ret i32 0
}

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress noreturn uwtable
define dso_local void @_Z7rethrowv() local_unnamed_addr #2 {
  tail call void @__cxa_rethrow() #9
  unreachable
}

declare void @__cxa_rethrow() local_unnamed_addr

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 personality ptr @__gxx_personality_v0 {
  %1 = invoke noundef i32 @_Z6calleej(i32 noundef 0)
          to label %247 unwind label %2

2:                                                ; preds = %0
  %3 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %4 = extractvalue { ptr, i32 } %3, 0
  %5 = extractvalue { ptr, i32 } %3, 1
  %6 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %8, label %11

8:                                                ; preds = %2
  %9 = tail call ptr @__cxa_begin_catch(ptr %4) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %17

10:                                               ; preds = %231, %207, %183, %159, %135, %111, %87, %63, %39, %8
  unreachable

11:                                               ; preds = %2
  %12 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %13 = icmp eq i32 %5, %12
  %14 = tail call ptr @__cxa_begin_catch(ptr %4) #8
  %15 = select i1 %13, ptr @.str.1, ptr @.str
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %15, i32 noundef 0)
  br label %25

17:                                               ; preds = %8
  %18 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %19 = extractvalue { ptr, i32 } %18, 1
  %20 = icmp eq i32 %19, %5
  br i1 %20, label %21, label %244

21:                                               ; preds = %17
  %22 = extractvalue { ptr, i32 } %18, 0
  %23 = tail call ptr @__cxa_begin_catch(ptr %22) #8
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 0)
  invoke void @__cxa_end_catch()
          to label %25 unwind label %242

25:                                               ; preds = %11, %21
  tail call void @__cxa_end_catch()
  %26 = invoke noundef i32 @_Z6calleej(i32 noundef 1)
          to label %247 unwind label %27

27:                                               ; preds = %25
  %28 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %29 = extractvalue { ptr, i32 } %28, 0
  %30 = extractvalue { ptr, i32 } %28, 1
  %31 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %32 = icmp eq i32 %30, %31
  br i1 %32, label %39, label %33

33:                                               ; preds = %27
  %34 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %35 = icmp eq i32 %30, %34
  %36 = tail call ptr @__cxa_begin_catch(ptr %29) #8
  %37 = select i1 %35, ptr @.str.1, ptr @.str
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %37, i32 noundef 1)
  br label %49

39:                                               ; preds = %27
  %40 = tail call ptr @__cxa_begin_catch(ptr %29) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %41

41:                                               ; preds = %39
  %42 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %43 = extractvalue { ptr, i32 } %42, 1
  %44 = icmp eq i32 %43, %30
  br i1 %44, label %45, label %244

45:                                               ; preds = %41
  %46 = extractvalue { ptr, i32 } %42, 0
  %47 = tail call ptr @__cxa_begin_catch(ptr %46) #8
  %48 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 1)
  invoke void @__cxa_end_catch()
          to label %49 unwind label %242

49:                                               ; preds = %45, %33
  tail call void @__cxa_end_catch()
  %50 = invoke noundef i32 @_Z6calleej(i32 noundef 2)
          to label %247 unwind label %51

51:                                               ; preds = %49
  %52 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %53 = extractvalue { ptr, i32 } %52, 0
  %54 = extractvalue { ptr, i32 } %52, 1
  %55 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %56 = icmp eq i32 %54, %55
  br i1 %56, label %63, label %57

57:                                               ; preds = %51
  %58 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %59 = icmp eq i32 %54, %58
  %60 = tail call ptr @__cxa_begin_catch(ptr %53) #8
  %61 = select i1 %59, ptr @.str.1, ptr @.str
  %62 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %61, i32 noundef 2)
  br label %73

63:                                               ; preds = %51
  %64 = tail call ptr @__cxa_begin_catch(ptr %53) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %65

65:                                               ; preds = %63
  %66 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %67 = extractvalue { ptr, i32 } %66, 1
  %68 = icmp eq i32 %67, %54
  br i1 %68, label %69, label %244

69:                                               ; preds = %65
  %70 = extractvalue { ptr, i32 } %66, 0
  %71 = tail call ptr @__cxa_begin_catch(ptr %70) #8
  %72 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 2)
  invoke void @__cxa_end_catch()
          to label %73 unwind label %242

73:                                               ; preds = %69, %57
  tail call void @__cxa_end_catch()
  %74 = invoke noundef i32 @_Z6calleej(i32 noundef 3)
          to label %247 unwind label %75

75:                                               ; preds = %73
  %76 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %77 = extractvalue { ptr, i32 } %76, 0
  %78 = extractvalue { ptr, i32 } %76, 1
  %79 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %80 = icmp eq i32 %78, %79
  br i1 %80, label %87, label %81

81:                                               ; preds = %75
  %82 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %83 = icmp eq i32 %78, %82
  %84 = tail call ptr @__cxa_begin_catch(ptr %77) #8
  %85 = select i1 %83, ptr @.str.1, ptr @.str
  %86 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %85, i32 noundef 3)
  br label %97

87:                                               ; preds = %75
  %88 = tail call ptr @__cxa_begin_catch(ptr %77) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %89

89:                                               ; preds = %87
  %90 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %91 = extractvalue { ptr, i32 } %90, 1
  %92 = icmp eq i32 %91, %78
  br i1 %92, label %93, label %244

93:                                               ; preds = %89
  %94 = extractvalue { ptr, i32 } %90, 0
  %95 = tail call ptr @__cxa_begin_catch(ptr %94) #8
  %96 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 3)
  invoke void @__cxa_end_catch()
          to label %97 unwind label %242

97:                                               ; preds = %93, %81
  tail call void @__cxa_end_catch()
  %98 = invoke noundef i32 @_Z6calleej(i32 noundef 4)
          to label %247 unwind label %99

99:                                               ; preds = %97
  %100 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %101 = extractvalue { ptr, i32 } %100, 0
  %102 = extractvalue { ptr, i32 } %100, 1
  %103 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %104 = icmp eq i32 %102, %103
  br i1 %104, label %111, label %105

105:                                              ; preds = %99
  %106 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %107 = icmp eq i32 %102, %106
  %108 = tail call ptr @__cxa_begin_catch(ptr %101) #8
  %109 = select i1 %107, ptr @.str.1, ptr @.str
  %110 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %109, i32 noundef 4)
  br label %121

111:                                              ; preds = %99
  %112 = tail call ptr @__cxa_begin_catch(ptr %101) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %113

113:                                              ; preds = %111
  %114 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %115 = extractvalue { ptr, i32 } %114, 1
  %116 = icmp eq i32 %115, %102
  br i1 %116, label %117, label %244

117:                                              ; preds = %113
  %118 = extractvalue { ptr, i32 } %114, 0
  %119 = tail call ptr @__cxa_begin_catch(ptr %118) #8
  %120 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 4)
  invoke void @__cxa_end_catch()
          to label %121 unwind label %242

121:                                              ; preds = %117, %105
  tail call void @__cxa_end_catch()
  %122 = invoke noundef i32 @_Z6calleej(i32 noundef 5)
          to label %247 unwind label %123

123:                                              ; preds = %121
  %124 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %125 = extractvalue { ptr, i32 } %124, 0
  %126 = extractvalue { ptr, i32 } %124, 1
  %127 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %128 = icmp eq i32 %126, %127
  br i1 %128, label %135, label %129

129:                                              ; preds = %123
  %130 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %131 = icmp eq i32 %126, %130
  %132 = tail call ptr @__cxa_begin_catch(ptr %125) #8
  %133 = select i1 %131, ptr @.str.1, ptr @.str
  %134 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %133, i32 noundef 5)
  br label %145

135:                                              ; preds = %123
  %136 = tail call ptr @__cxa_begin_catch(ptr %125) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %137

137:                                              ; preds = %135
  %138 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %139 = extractvalue { ptr, i32 } %138, 1
  %140 = icmp eq i32 %139, %126
  br i1 %140, label %141, label %244

141:                                              ; preds = %137
  %142 = extractvalue { ptr, i32 } %138, 0
  %143 = tail call ptr @__cxa_begin_catch(ptr %142) #8
  %144 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 5)
  invoke void @__cxa_end_catch()
          to label %145 unwind label %242

145:                                              ; preds = %141, %129
  tail call void @__cxa_end_catch()
  %146 = invoke noundef i32 @_Z6calleej(i32 noundef 6)
          to label %247 unwind label %147

147:                                              ; preds = %145
  %148 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %149 = extractvalue { ptr, i32 } %148, 0
  %150 = extractvalue { ptr, i32 } %148, 1
  %151 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %152 = icmp eq i32 %150, %151
  br i1 %152, label %159, label %153

153:                                              ; preds = %147
  %154 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %155 = icmp eq i32 %150, %154
  %156 = tail call ptr @__cxa_begin_catch(ptr %149) #8
  %157 = select i1 %155, ptr @.str.1, ptr @.str
  %158 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %157, i32 noundef 6)
  br label %169

159:                                              ; preds = %147
  %160 = tail call ptr @__cxa_begin_catch(ptr %149) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %161

161:                                              ; preds = %159
  %162 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %163 = extractvalue { ptr, i32 } %162, 1
  %164 = icmp eq i32 %163, %150
  br i1 %164, label %165, label %244

165:                                              ; preds = %161
  %166 = extractvalue { ptr, i32 } %162, 0
  %167 = tail call ptr @__cxa_begin_catch(ptr %166) #8
  %168 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 6)
  invoke void @__cxa_end_catch()
          to label %169 unwind label %242

169:                                              ; preds = %165, %153
  tail call void @__cxa_end_catch()
  %170 = invoke noundef i32 @_Z6calleej(i32 noundef 7)
          to label %247 unwind label %171

171:                                              ; preds = %169
  %172 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %173 = extractvalue { ptr, i32 } %172, 0
  %174 = extractvalue { ptr, i32 } %172, 1
  %175 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %176 = icmp eq i32 %174, %175
  br i1 %176, label %183, label %177

177:                                              ; preds = %171
  %178 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %179 = icmp eq i32 %174, %178
  %180 = tail call ptr @__cxa_begin_catch(ptr %173) #8
  %181 = select i1 %179, ptr @.str.1, ptr @.str
  %182 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %181, i32 noundef 7)
  br label %193

183:                                              ; preds = %171
  %184 = tail call ptr @__cxa_begin_catch(ptr %173) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %185

185:                                              ; preds = %183
  %186 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %187 = extractvalue { ptr, i32 } %186, 1
  %188 = icmp eq i32 %187, %174
  br i1 %188, label %189, label %244

189:                                              ; preds = %185
  %190 = extractvalue { ptr, i32 } %186, 0
  %191 = tail call ptr @__cxa_begin_catch(ptr %190) #8
  %192 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 7)
  invoke void @__cxa_end_catch()
          to label %193 unwind label %242

193:                                              ; preds = %189, %177
  tail call void @__cxa_end_catch()
  %194 = invoke noundef i32 @_Z6calleej(i32 noundef 8)
          to label %247 unwind label %195

195:                                              ; preds = %193
  %196 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %197 = extractvalue { ptr, i32 } %196, 0
  %198 = extractvalue { ptr, i32 } %196, 1
  %199 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %200 = icmp eq i32 %198, %199
  br i1 %200, label %207, label %201

201:                                              ; preds = %195
  %202 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %203 = icmp eq i32 %198, %202
  %204 = tail call ptr @__cxa_begin_catch(ptr %197) #8
  %205 = select i1 %203, ptr @.str.1, ptr @.str
  %206 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %205, i32 noundef 8)
  br label %217

207:                                              ; preds = %195
  %208 = tail call ptr @__cxa_begin_catch(ptr %197) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %209

209:                                              ; preds = %207
  %210 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %211 = extractvalue { ptr, i32 } %210, 1
  %212 = icmp eq i32 %211, %198
  br i1 %212, label %213, label %244

213:                                              ; preds = %209
  %214 = extractvalue { ptr, i32 } %210, 0
  %215 = tail call ptr @__cxa_begin_catch(ptr %214) #8
  %216 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 8)
  invoke void @__cxa_end_catch()
          to label %217 unwind label %242

217:                                              ; preds = %213, %201
  tail call void @__cxa_end_catch()
  %218 = invoke noundef i32 @_Z6calleej(i32 noundef 9)
          to label %247 unwind label %219

219:                                              ; preds = %217
  %220 = landingpad { ptr, i32 }
          catch ptr @_ZTI3foo
          catch ptr @_ZTIi
          catch ptr null
  %221 = extractvalue { ptr, i32 } %220, 0
  %222 = extractvalue { ptr, i32 } %220, 1
  %223 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI3foo) #8
  %224 = icmp eq i32 %222, %223
  br i1 %224, label %231, label %225

225:                                              ; preds = %219
  %226 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %227 = icmp eq i32 %222, %226
  %228 = tail call ptr @__cxa_begin_catch(ptr %221) #8
  %229 = select i1 %227, ptr @.str.1, ptr @.str
  %230 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %229, i32 noundef 9)
  br label %241

231:                                              ; preds = %219
  %232 = tail call ptr @__cxa_begin_catch(ptr %221) #8
  invoke void @__cxa_rethrow() #9
          to label %10 unwind label %233

233:                                              ; preds = %231
  %234 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI3foo
  %235 = extractvalue { ptr, i32 } %234, 1
  %236 = icmp eq i32 %235, %222
  br i1 %236, label %237, label %244

237:                                              ; preds = %233
  %238 = extractvalue { ptr, i32 } %234, 0
  %239 = tail call ptr @__cxa_begin_catch(ptr %238) #8
  %240 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 9)
  invoke void @__cxa_end_catch()
          to label %241 unwind label %242

241:                                              ; preds = %237, %225
  tail call void @__cxa_end_catch()
  br label %247

242:                                              ; preds = %237, %213, %189, %165, %141, %117, %93, %69, %45, %21
  %243 = landingpad { ptr, i32 }
          cleanup
  br label %244

244:                                              ; preds = %17, %41, %65, %89, %113, %137, %161, %185, %209, %233, %242
  %245 = phi { ptr, i32 } [ %243, %242 ], [ %18, %17 ], [ %42, %41 ], [ %66, %65 ], [ %90, %89 ], [ %114, %113 ], [ %138, %137 ], [ %162, %161 ], [ %186, %185 ], [ %210, %209 ], [ %234, %233 ]
  invoke void @__cxa_end_catch()
          to label %246 unwind label %248

246:                                              ; preds = %244
  resume { ptr, i32 } %245

247:                                              ; preds = %241, %217, %193, %169, %145, %121, %97, %73, %49, %25, %0
  ret i32 0

248:                                              ; preds = %244
  %249 = landingpad { ptr, i32 }
          catch ptr null
  %250 = extractvalue { ptr, i32 } %249, 0
  tail call void @__clang_call_terminate(ptr %250) #10
  unreachable
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #4

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #5

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #6 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #8
  tail call void @_ZSt9terminatev() #10
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #7

attributes #0 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold noreturn }
attributes #2 = { mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nosync nounwind memory(none) }
attributes #5 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold nofree noreturn }
attributes #8 = { nounwind }
attributes #9 = { noreturn }
attributes #10 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
!12 = !{!13, !7, i64 0}
!13 = !{!"_ZTS3foo", !7, i64 0}
