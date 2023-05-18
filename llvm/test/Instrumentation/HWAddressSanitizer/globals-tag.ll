; RUN: opt < %s -S -passes=hwasan -mtriple=aarch64-unknown-linux | FileCheck %s --implicit-check-not '\00\00\00\00\00\00\00'

source_filename = "test.cpp"

; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00 "
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00!"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00#"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00%"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00&"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00'"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00("
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00)"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00*"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00+"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00,"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00-"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00."
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00/"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00:"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00;"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00<"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00="
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00>"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00?"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00@"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00["
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\\"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00]"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00^"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00_"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00`"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00{"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00|"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00}"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00~"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00$"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\000"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\001"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\10"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\11"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\12"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\13"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\14"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\15"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\16"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\17"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\18"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\19"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\1A"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\1B"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\1C"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\1D"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\1E"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\1F"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\002"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\22"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\003"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\004"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\005"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\006"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\007"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\7F"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\008"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\80"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\81"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\82"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\83"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\84"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\85"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\86"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\87"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\88"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\89"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\8A"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\8B"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\8C"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\8D"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\8E"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\8F"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\009"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\90"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\91"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\92"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\93"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\94"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\95"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\96"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\97"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\98"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\99"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\9A"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\9B"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\9C"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\9D"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\9E"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\9F"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00a"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00A"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A0"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A1"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A2"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A3"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A4"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A5"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A6"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A7"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A8"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\A9"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\AA"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\AB"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\AC"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\AD"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\AE"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\AF"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00b"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00B"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B0"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B1"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B2"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B3"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B4"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B5"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B6"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B7"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B8"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\B9"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\BA"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\BB"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\BC"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\BD"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\BE"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\BF"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00c"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00C"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C0"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C1"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C2"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C3"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C4"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C5"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C6"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C7"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C8"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\C9"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\CA"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\CB"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\CC"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\CD"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\CE"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\CF"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00d"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00D"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D0"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D1"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D2"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D3"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D4"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D5"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D6"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D7"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D8"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\D9"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\DA"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\DB"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\DC"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\DD"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\DE"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\DF"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\DF"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00e"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00E"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E0"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E0"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E1"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E1"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E2"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E2"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E3"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E3"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E4"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E4"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E5"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E5"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E6"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E6"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E7"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E7"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E8"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E8"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E9"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\E9"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\EA"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\EA"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\EB"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\EB"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\EC"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\EC"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\ED"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\ED"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\EE"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\EE"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\EF"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00f"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00F"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F0"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F1"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F2"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F3"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F4"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F5"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F6"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F7"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F8"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\F9"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\FA"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\FB"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\FC"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\FD"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\FE"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00\FF"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00g"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00G"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00h"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00H"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00i"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00I"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00j"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00J"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00k"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00K"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00l"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00L"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00m"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00M"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00n"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00N"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00o"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00O"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00p"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00P"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00q"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00Q"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00r"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00R"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00s"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00S"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00t"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00T"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00u"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00U"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00v"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00V"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00w"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00W"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00x"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00X"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00y"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00Y"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00z"
; CHECK-DAG: "\00\00\00\00\00\00\00\00\00\00\00Z"

@g_0 = dso_local global i32 0, align 4
@g_1 = dso_local global i32 0, align 4
@g_2 = dso_local global i32 0, align 4
@g_3 = dso_local global i32 0, align 4
@g_4 = dso_local global i32 0, align 4
@g_5 = dso_local global i32 0, align 4
@g_6 = dso_local global i32 0, align 4
@g_7 = dso_local global i32 0, align 4
@g_8 = dso_local global i32 0, align 4
@g_9 = dso_local global i32 0, align 4
@g_10 = dso_local global i32 0, align 4
@g_11 = dso_local global i32 0, align 4
@g_12 = dso_local global i32 0, align 4
@g_13 = dso_local global i32 0, align 4
@g_14 = dso_local global i32 0, align 4
@g_15 = dso_local global i32 0, align 4
@g_16 = dso_local global i32 0, align 4
@g_17 = dso_local global i32 0, align 4
@g_18 = dso_local global i32 0, align 4
@g_19 = dso_local global i32 0, align 4
@g_20 = dso_local global i32 0, align 4
@g_21 = dso_local global i32 0, align 4
@g_22 = dso_local global i32 0, align 4
@g_23 = dso_local global i32 0, align 4
@g_24 = dso_local global i32 0, align 4
@g_25 = dso_local global i32 0, align 4
@g_26 = dso_local global i32 0, align 4
@g_27 = dso_local global i32 0, align 4
@g_28 = dso_local global i32 0, align 4
@g_29 = dso_local global i32 0, align 4
@g_30 = dso_local global i32 0, align 4
@g_31 = dso_local global i32 0, align 4
@g_32 = dso_local global i32 0, align 4
@g_33 = dso_local global i32 0, align 4
@g_34 = dso_local global i32 0, align 4
@g_35 = dso_local global i32 0, align 4
@g_36 = dso_local global i32 0, align 4
@g_37 = dso_local global i32 0, align 4
@g_38 = dso_local global i32 0, align 4
@g_39 = dso_local global i32 0, align 4
@g_40 = dso_local global i32 0, align 4
@g_41 = dso_local global i32 0, align 4
@g_42 = dso_local global i32 0, align 4
@g_43 = dso_local global i32 0, align 4
@g_44 = dso_local global i32 0, align 4
@g_45 = dso_local global i32 0, align 4
@g_46 = dso_local global i32 0, align 4
@g_47 = dso_local global i32 0, align 4
@g_48 = dso_local global i32 0, align 4
@g_49 = dso_local global i32 0, align 4
@g_50 = dso_local global i32 0, align 4
@g_51 = dso_local global i32 0, align 4
@g_52 = dso_local global i32 0, align 4
@g_53 = dso_local global i32 0, align 4
@g_54 = dso_local global i32 0, align 4
@g_55 = dso_local global i32 0, align 4
@g_56 = dso_local global i32 0, align 4
@g_57 = dso_local global i32 0, align 4
@g_58 = dso_local global i32 0, align 4
@g_59 = dso_local global i32 0, align 4
@g_60 = dso_local global i32 0, align 4
@g_61 = dso_local global i32 0, align 4
@g_62 = dso_local global i32 0, align 4
@g_63 = dso_local global i32 0, align 4
@g_64 = dso_local global i32 0, align 4
@g_65 = dso_local global i32 0, align 4
@g_66 = dso_local global i32 0, align 4
@g_67 = dso_local global i32 0, align 4
@g_68 = dso_local global i32 0, align 4
@g_69 = dso_local global i32 0, align 4
@g_70 = dso_local global i32 0, align 4
@g_71 = dso_local global i32 0, align 4
@g_72 = dso_local global i32 0, align 4
@g_73 = dso_local global i32 0, align 4
@g_74 = dso_local global i32 0, align 4
@g_75 = dso_local global i32 0, align 4
@g_76 = dso_local global i32 0, align 4
@g_77 = dso_local global i32 0, align 4
@g_78 = dso_local global i32 0, align 4
@g_79 = dso_local global i32 0, align 4
@g_80 = dso_local global i32 0, align 4
@g_81 = dso_local global i32 0, align 4
@g_82 = dso_local global i32 0, align 4
@g_83 = dso_local global i32 0, align 4
@g_84 = dso_local global i32 0, align 4
@g_85 = dso_local global i32 0, align 4
@g_86 = dso_local global i32 0, align 4
@g_87 = dso_local global i32 0, align 4
@g_88 = dso_local global i32 0, align 4
@g_89 = dso_local global i32 0, align 4
@g_90 = dso_local global i32 0, align 4
@g_91 = dso_local global i32 0, align 4
@g_92 = dso_local global i32 0, align 4
@g_93 = dso_local global i32 0, align 4
@g_94 = dso_local global i32 0, align 4
@g_95 = dso_local global i32 0, align 4
@g_96 = dso_local global i32 0, align 4
@g_97 = dso_local global i32 0, align 4
@g_98 = dso_local global i32 0, align 4
@g_99 = dso_local global i32 0, align 4
@g_100 = dso_local global i32 0, align 4
@g_101 = dso_local global i32 0, align 4
@g_102 = dso_local global i32 0, align 4
@g_103 = dso_local global i32 0, align 4
@g_104 = dso_local global i32 0, align 4
@g_105 = dso_local global i32 0, align 4
@g_106 = dso_local global i32 0, align 4
@g_107 = dso_local global i32 0, align 4
@g_108 = dso_local global i32 0, align 4
@g_109 = dso_local global i32 0, align 4
@g_110 = dso_local global i32 0, align 4
@g_111 = dso_local global i32 0, align 4
@g_112 = dso_local global i32 0, align 4
@g_113 = dso_local global i32 0, align 4
@g_114 = dso_local global i32 0, align 4
@g_115 = dso_local global i32 0, align 4
@g_116 = dso_local global i32 0, align 4
@g_117 = dso_local global i32 0, align 4
@g_118 = dso_local global i32 0, align 4
@g_119 = dso_local global i32 0, align 4
@g_120 = dso_local global i32 0, align 4
@g_121 = dso_local global i32 0, align 4
@g_122 = dso_local global i32 0, align 4
@g_123 = dso_local global i32 0, align 4
@g_124 = dso_local global i32 0, align 4
@g_125 = dso_local global i32 0, align 4
@g_126 = dso_local global i32 0, align 4
@g_127 = dso_local global i32 0, align 4
@g_128 = dso_local global i32 0, align 4
@g_129 = dso_local global i32 0, align 4
@g_130 = dso_local global i32 0, align 4
@g_131 = dso_local global i32 0, align 4
@g_132 = dso_local global i32 0, align 4
@g_133 = dso_local global i32 0, align 4
@g_134 = dso_local global i32 0, align 4
@g_135 = dso_local global i32 0, align 4
@g_136 = dso_local global i32 0, align 4
@g_137 = dso_local global i32 0, align 4
@g_138 = dso_local global i32 0, align 4
@g_139 = dso_local global i32 0, align 4
@g_140 = dso_local global i32 0, align 4
@g_141 = dso_local global i32 0, align 4
@g_142 = dso_local global i32 0, align 4
@g_143 = dso_local global i32 0, align 4
@g_144 = dso_local global i32 0, align 4
@g_145 = dso_local global i32 0, align 4
@g_146 = dso_local global i32 0, align 4
@g_147 = dso_local global i32 0, align 4
@g_148 = dso_local global i32 0, align 4
@g_149 = dso_local global i32 0, align 4
@g_150 = dso_local global i32 0, align 4
@g_151 = dso_local global i32 0, align 4
@g_152 = dso_local global i32 0, align 4
@g_153 = dso_local global i32 0, align 4
@g_154 = dso_local global i32 0, align 4
@g_155 = dso_local global i32 0, align 4
@g_156 = dso_local global i32 0, align 4
@g_157 = dso_local global i32 0, align 4
@g_158 = dso_local global i32 0, align 4
@g_159 = dso_local global i32 0, align 4
@g_160 = dso_local global i32 0, align 4
@g_161 = dso_local global i32 0, align 4
@g_162 = dso_local global i32 0, align 4
@g_163 = dso_local global i32 0, align 4
@g_164 = dso_local global i32 0, align 4
@g_165 = dso_local global i32 0, align 4
@g_166 = dso_local global i32 0, align 4
@g_167 = dso_local global i32 0, align 4
@g_168 = dso_local global i32 0, align 4
@g_169 = dso_local global i32 0, align 4
@g_170 = dso_local global i32 0, align 4
@g_171 = dso_local global i32 0, align 4
@g_172 = dso_local global i32 0, align 4
@g_173 = dso_local global i32 0, align 4
@g_174 = dso_local global i32 0, align 4
@g_175 = dso_local global i32 0, align 4
@g_176 = dso_local global i32 0, align 4
@g_177 = dso_local global i32 0, align 4
@g_178 = dso_local global i32 0, align 4
@g_179 = dso_local global i32 0, align 4
@g_180 = dso_local global i32 0, align 4
@g_181 = dso_local global i32 0, align 4
@g_182 = dso_local global i32 0, align 4
@g_183 = dso_local global i32 0, align 4
@g_184 = dso_local global i32 0, align 4
@g_185 = dso_local global i32 0, align 4
@g_186 = dso_local global i32 0, align 4
@g_187 = dso_local global i32 0, align 4
@g_188 = dso_local global i32 0, align 4
@g_189 = dso_local global i32 0, align 4
@g_190 = dso_local global i32 0, align 4
@g_191 = dso_local global i32 0, align 4
@g_192 = dso_local global i32 0, align 4
@g_193 = dso_local global i32 0, align 4
@g_194 = dso_local global i32 0, align 4
@g_195 = dso_local global i32 0, align 4
@g_196 = dso_local global i32 0, align 4
@g_197 = dso_local global i32 0, align 4
@g_198 = dso_local global i32 0, align 4
@g_199 = dso_local global i32 0, align 4
@g_200 = dso_local global i32 0, align 4
@g_201 = dso_local global i32 0, align 4
@g_202 = dso_local global i32 0, align 4
@g_203 = dso_local global i32 0, align 4
@g_204 = dso_local global i32 0, align 4
@g_205 = dso_local global i32 0, align 4
@g_206 = dso_local global i32 0, align 4
@g_207 = dso_local global i32 0, align 4
@g_208 = dso_local global i32 0, align 4
@g_209 = dso_local global i32 0, align 4
@g_210 = dso_local global i32 0, align 4
@g_211 = dso_local global i32 0, align 4
@g_212 = dso_local global i32 0, align 4
@g_213 = dso_local global i32 0, align 4
@g_214 = dso_local global i32 0, align 4
@g_215 = dso_local global i32 0, align 4
@g_216 = dso_local global i32 0, align 4
@g_217 = dso_local global i32 0, align 4
@g_218 = dso_local global i32 0, align 4
@g_219 = dso_local global i32 0, align 4
@g_220 = dso_local global i32 0, align 4
@g_221 = dso_local global i32 0, align 4
@g_222 = dso_local global i32 0, align 4
@g_223 = dso_local global i32 0, align 4
@g_224 = dso_local global i32 0, align 4
@g_225 = dso_local global i32 0, align 4
@g_226 = dso_local global i32 0, align 4
@g_227 = dso_local global i32 0, align 4
@g_228 = dso_local global i32 0, align 4
@g_229 = dso_local global i32 0, align 4
@g_230 = dso_local global i32 0, align 4
@g_231 = dso_local global i32 0, align 4
@g_232 = dso_local global i32 0, align 4
@g_233 = dso_local global i32 0, align 4
@g_234 = dso_local global i32 0, align 4
@g_235 = dso_local global i32 0, align 4
@g_236 = dso_local global i32 0, align 4
@g_237 = dso_local global i32 0, align 4
@g_238 = dso_local global i32 0, align 4
@g_239 = dso_local global i32 0, align 4
@g_240 = dso_local global i32 0, align 4
@g_241 = dso_local global i32 0, align 4
@g_242 = dso_local global i32 0, align 4
@g_243 = dso_local global i32 0, align 4
@g_244 = dso_local global i32 0, align 4
@g_245 = dso_local global i32 0, align 4
@g_246 = dso_local global i32 0, align 4
@g_247 = dso_local global i32 0, align 4
@g_248 = dso_local global i32 0, align 4
@g_249 = dso_local global i32 0, align 4
@g_250 = dso_local global i32 0, align 4
@g_251 = dso_local global i32 0, align 4
@g_252 = dso_local global i32 0, align 4
@g_253 = dso_local global i32 0, align 4
@g_254 = dso_local global i32 0, align 4
@g_255 = dso_local global i32 0, align 4
