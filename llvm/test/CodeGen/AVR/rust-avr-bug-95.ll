; RUN: llc < %s -mtriple=avr | FileCheck %s

%"fmt::Formatter.1.77.153.229.305.381.1673" = type { [0 x i8], i32, [0 x i8], i32, [0 x i8], i8, [0 x i8], %"option::Option<usize>.0.76.152.228.304.380.1672", [0 x i8], %"option::Option<usize>.0.76.152.228.304.380.1672", [0 x i8], { ptr, ptr }, [0 x i8], { ptr, ptr }, [0 x i8], { ptr, i16 }, [0 x i8] }
%"option::Option<usize>.0.76.152.228.304.380.1672" = type { [0 x i8], i8, [2 x i8] }

@str.4S = external constant [5 x i8]

; Function Attrs: uwtable
define void @"_ZN65_$LT$lib..str..Chars$LT$$u27$a$GT$$u20$as$u20$lib..fmt..Debug$GT$3fmt17h76a537e22649f739E"(ptr dereferenceable(27) %__arg_0) unnamed_addr #0 personality ptr addrspace(1) @rust_eh_personality {
; CHECK-LABEL: "_ZN65_$LT$lib..str..Chars$LT$$u27$a$GT$$u20$as$u20$lib..fmt..Debug$GT$3fmt17h76a537e22649f739E"
start:
  %0 = getelementptr inbounds %"fmt::Formatter.1.77.153.229.305.381.1673", ptr %__arg_0, i16 0, i32 11, i32 0
  %1 = load ptr, ptr %0, align 1, !noalias !0, !nonnull !9
  %2 = getelementptr inbounds %"fmt::Formatter.1.77.153.229.305.381.1673", ptr %__arg_0, i16 0, i32 11, i32 1
  %3 = bitcast ptr %2 to ptr
  %4 = load ptr, ptr %3, align 1, !noalias !0, !nonnull !9
  %5 = getelementptr inbounds ptr addrspace(1), ptr %4, i16 3
  %6 = load ptr addrspace(1), ptr %5, align 1, !invariant.load !9, !noalias !0, !nonnull !9
  %7 = tail call zeroext i1 %6(ptr nonnull %1, ptr noalias nonnull readonly @str.4S, i16 5), !noalias !10
  unreachable
}

declare i32 @rust_eh_personality(...) unnamed_addr

attributes #0 = { uwtable }

!0 = !{!1, !3, !5, !6, !8}
!1 = distinct !{!1, !2, !"_ZN3lib3fmt9Formatter9write_str17ha1a9656fc66ccbe5E: %data.0"}
!2 = distinct !{!2, !"_ZN3lib3fmt9Formatter9write_str17ha1a9656fc66ccbe5E"}
!3 = distinct !{!3, !4, !"_ZN3lib3fmt8builders16debug_struct_new17h352a1de8f89c2bc3E: argument 0"}
!4 = distinct !{!4, !"_ZN3lib3fmt8builders16debug_struct_new17h352a1de8f89c2bc3E"}
!5 = distinct !{!5, !4, !"_ZN3lib3fmt8builders16debug_struct_new17h352a1de8f89c2bc3E: %name.0"}
!6 = distinct !{!6, !7, !"_ZN3lib3fmt9Formatter12debug_struct17ha1ff79f633171b68E: argument 0"}
!7 = distinct !{!7, !"_ZN3lib3fmt9Formatter12debug_struct17ha1ff79f633171b68E"}
!8 = distinct !{!8, !7, !"_ZN3lib3fmt9Formatter12debug_struct17ha1ff79f633171b68E: %name.0"}
!9 = !{}
!10 = !{!3, !6}
