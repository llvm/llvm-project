; RUN: llc -verify-machineinstrs < %s -march=ppc64 -mtriple=powerpc64-unknown-linux-gnu
; RUN: llc -verify-machineinstrs < %s -march=ppc64le -mtriple=powerpc64le-unknown-linux-gnu

%"class.std::__1::locale::id.1580.4307.4610.8491" = type { %"struct.std::__1::once_flag.1579.4306.4609.8490", i32 }
%"struct.std::__1::once_flag.1579.4306.4609.8490" = type { i64 }
%"class.Foam::IOerror.1581.4308.4611.8505" = type { %"class.Foam::error.1535.4262.4565.8504", %"class.Foam::string.1530.4257.4560.8499", i32, i32 }
%"class.Foam::error.1535.4262.4565.8504" = type { %"class.std::exception.1523.4250.4553.8492", [36 x i8], %"class.Foam::string.1530.4257.4560.8499", %"class.Foam::string.1530.4257.4560.8499", i32, i8, i8, %"class.Foam::OStringStream.1534.4261.4564.8503"* }
%"class.std::exception.1523.4250.4553.8492" = type { i32 (...)** }
%"class.Foam::OStringStream.1534.4261.4564.8503" = type { %"class.Foam::OSstream.1533.4260.4563.8502" }
%"class.Foam::OSstream.1533.4260.4563.8502" = type { [50 x i8], %"class.Foam::fileName.1531.4258.4561.8500", %"class.std::__1::basic_ostream.1532.4259.4562.8501"* }
%"class.Foam::fileName.1531.4258.4561.8500" = type { %"class.Foam::string.1530.4257.4560.8499" }
%"class.std::__1::basic_ostream.1532.4259.4562.8501" = type { i32 (...)**, [148 x i8] }
%"class.Foam::string.1530.4257.4560.8499" = type { %"class.std::__1::basic_string.1529.4256.4559.8498" }
%"class.std::__1::basic_string.1529.4256.4559.8498" = type { %"class.std::__1::__compressed_pair.1528.4255.4558.8497" }
%"class.std::__1::__compressed_pair.1528.4255.4558.8497" = type { %"class.std::__1::__libcpp_compressed_pair_imp.1527.4254.4557.8496" }
%"class.std::__1::__libcpp_compressed_pair_imp.1527.4254.4557.8496" = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep.1526.4253.4556.8495" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep.1526.4253.4556.8495" = type { %union.anon.1525.4252.4555.8494 }
%union.anon.1525.4252.4555.8494 = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long.1524.4251.4554.8493" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long.1524.4251.4554.8493" = type { i64, i64, i8* }

@.str3 = external unnamed_addr constant [16 x i8], align 1
@_ZNSt3__15ctypeIcE2idE = external global %"class.std::__1::locale::id.1580.4307.4610.8491"
@_ZN4Foam12FatalIOErrorE = external global %"class.Foam::IOerror.1581.4308.4611.8505"
@.str204 = external unnamed_addr constant [18 x i8], align 1
@.str205 = external unnamed_addr constant [34 x i8], align 1

declare void @_ZN4FoamlsERNS_7OstreamEPKc() #0

declare i32 @__gxx_personality_v0(...)

declare void @_ZNKSt3__18ios_base6getlocEv() #0

declare void @_ZNKSt3__16locale9use_facetERNS0_2idE() #0

; Function Attrs: noreturn
declare void @_ZNKSt3__121__basic_string_commonILb1EE20__throw_length_errorEv() #1 align 2

declare void @_ZN4Foam6string6expandEb() #0

declare void @_ZN4Foam8IFstreamC1ERKNS_8fileNameENS_8IOstream12streamFormatENS4_13versionNumberE() #0

declare void @_ZN4Foam7IOerrorclEPKcS2_iRKNS_8IOstreamE() #0

declare void @_ZN4Foam7IOerror4exitEi() #0

; Function Attrs: inlinehint
declare void @_ZN4Foam8fileName12stripInvalidEv() #2 align 2

define void @_ZN4Foam3CSVINS_6VectorIdEEE4readEv() #0 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @_ZN4Foam6string6expandEb()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  br i1 undef, label %if.then.i.i.i.i176, label %_ZN4Foam6stringC2ERKS0_.exit.i

if.then.i.i.i.i176:                               ; preds = %invoke.cont
  invoke void @_ZNKSt3__121__basic_string_commonILb1EE20__throw_length_errorEv()
          to label %.noexc unwind label %lpad

.noexc:                                           ; preds = %if.then.i.i.i.i176
  unreachable

_ZN4Foam6stringC2ERKS0_.exit.i:                   ; preds = %invoke.cont
  invoke void @_ZN4Foam8fileName12stripInvalidEv()
          to label %invoke.cont2 unwind label %lpad.i

lpad.i:                                           ; preds = %_ZN4Foam6stringC2ERKS0_.exit.i
  %0 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup142

invoke.cont2:                                     ; preds = %_ZN4Foam6stringC2ERKS0_.exit.i
  invoke void @_ZN4Foam8IFstreamC1ERKNS_8fileNameENS_8IOstream12streamFormatENS4_13versionNumberE()
          to label %invoke.cont4 unwind label %lpad3

invoke.cont4:                                     ; preds = %invoke.cont2
  br i1 undef, label %for.body, label %if.then

if.then:                                          ; preds = %invoke.cont4
  invoke void @_ZN4Foam7IOerrorclEPKcS2_iRKNS_8IOstreamE()
          to label %invoke.cont8 unwind label %lpad5

invoke.cont8:                                     ; preds = %if.then
  invoke void @_ZN4FoamlsERNS_7OstreamEPKc()
          to label %memptr.end.i unwind label %lpad5

memptr.end.i:                                     ; preds = %invoke.cont8
  invoke void @_ZN4Foam7IOerror4exitEi()
          to label %if.end unwind label %lpad5

lpad:                                             ; preds = %if.then.i.i.i.i176, %entry
  %1 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup142

lpad3:                                            ; preds = %invoke.cont2
  %2 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup142

lpad5:                                            ; preds = %memptr.end.i, %invoke.cont8, %if.then
  %3 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup142

if.end:                                           ; preds = %memptr.end.i
  br i1 undef, label %for.body, label %vector.body

for.body:                                         ; preds = %if.end, %invoke.cont4
  invoke void @_ZNKSt3__18ios_base6getlocEv()
          to label %.noexc205 unwind label %lpad19

.noexc205:                                        ; preds = %for.body
  invoke void @_ZNKSt3__16locale9use_facetERNS0_2idE()
          to label %invoke.cont.i.i.i unwind label %lpad.i.i.i

invoke.cont.i.i.i:                                ; preds = %.noexc205
  unreachable

lpad.i.i.i:                                       ; preds = %.noexc205
  %4 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup142

lpad19:                                           ; preds = %for.body
  %5 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup142

vector.body:                                      ; preds = %vector.body, %if.end
  %vec.phi = phi <8 x i32> [ %10, %vector.body ], [ undef, %if.end ]
  %vec.phi1302 = phi <8 x i32> [ %11, %vector.body ], [ undef, %if.end ]
  %vec.phi1303 = phi <8 x i32> [ %12, %vector.body ], [ undef, %if.end ]
  %vec.phi1304 = phi <8 x i32> [ %13, %vector.body ], [ undef, %if.end ]
  %6 = icmp sgt <8 x i32> undef, %vec.phi
  %7 = icmp sgt <8 x i32> undef, %vec.phi1302
  %8 = icmp sgt <8 x i32> undef, %vec.phi1303
  %9 = icmp sgt <8 x i32> undef, %vec.phi1304
  %10 = select <8 x i1> %6, <8 x i32> undef, <8 x i32> %vec.phi
  %11 = select <8 x i1> %7, <8 x i32> undef, <8 x i32> %vec.phi1302
  %12 = select <8 x i1> %8, <8 x i32> undef, <8 x i32> %vec.phi1303
  %13 = select <8 x i1> %9, <8 x i32> undef, <8 x i32> %vec.phi1304
  br label %vector.body

ehcleanup142:                                     ; preds = %lpad19, %lpad.i.i.i, %lpad5, %lpad3, %lpad, %lpad.i
  resume { i8*, i32 } undef
}

attributes #0 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inlinehint "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

