//===--- HexagonDepInstrFormats.td ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class Enc_12122225 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vx32;
  let Inst{7-3} = Vx32{4-0};
  bits <3> Qd8;
  let Inst{2-0} = Qd8{2-0};
}
class Enc_16626097 : OpcodeHexagon {
  bits <2> Qs4;
  let Inst{6-5} = Qs4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vw32;
  let Inst{4-0} = Vw32{4-0};
}
class Enc_13397056 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{10-8} = Ii{9-7};
  bits <2> Qv4;
  let Inst{12-11} = Qv4{1-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_7315939 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <6> n1;
  let Inst{28-28} = n1{5-5};
  let Inst{24-22} = n1{4-2};
  let Inst{13-13} = n1{1-1};
  let Inst{8-8} = n1{0-0};
}
class Enc_605928 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{13-13} = Ii{9-9};
  let Inst{10-8} = Ii{8-6};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Zdd8;
  let Inst{4-0} = Zdd8{4-0};
}
class Enc_15275738 : OpcodeHexagon {
  bits <12> Ii;
  let Inst{26-25} = Ii{11-10};
  let Inst{13-5} = Ii{9-1};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_12822813 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rxx32;
  let Inst{4-0} = Rxx32{4-0};
  bits <2> Pe4;
  let Inst{6-5} = Pe4{1-0};
}
class Enc_10282127 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{12-7} = Ii{6-1};
  bits <8> II;
  let Inst{13-13} = II{7-7};
  let Inst{6-0} = II{6-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_14264243 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <4> Rt16;
  let Inst{11-8} = Rt16{3-0};
}
class Enc_6778937 : OpcodeHexagon {
  bits <5> Rxx32;
  let Inst{20-16} = Rxx32{4-0};
  bits <0> sgp10;
}
class Enc_5480539 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <3> Rt8;
  let Inst{2-0} = Rt8{2-0};
  bits <5> Vxx32;
  let Inst{7-3} = Vxx32{4-0};
}
class Enc_11422009 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vy32;
  let Inst{12-8} = Vy32{4-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_16357011 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{8-4} = Vv32{4-0};
  bits <5> Vt32;
  let Inst{13-9} = Vt32{4-0};
  bits <4> Vdd16;
  let Inst{3-0} = Vdd16{3-0};
}
class Enc_4975051 : OpcodeHexagon {
  bits <19> Ii;
  let Inst{26-25} = Ii{18-17};
  let Inst{20-16} = Ii{16-12};
  let Inst{13-5} = Ii{11-3};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_14786238 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rtt32;
  let Inst{20-16} = Rtt32{4-0};
  bits <5> Vx32;
  let Inst{7-3} = Vx32{4-0};
}
class Enc_15472748 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_6773159 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{12-7} = Ii{5-0};
  bits <5> II;
  let Inst{4-0} = II{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_12535811 : OpcodeHexagon {
  bits <2> Qv4;
  let Inst{23-22} = Qv4{1-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_14007201 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-5} = Ii{7-0};
  bits <8> II;
  let Inst{22-16} = II{7-1};
  let Inst{13-13} = II{0-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_2577026 : OpcodeHexagon {
  bits <3> Qt8;
  let Inst{2-0} = Qt8{2-0};
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_7305764 : OpcodeHexagon {
  bits <5> II;
  let Inst{12-8} = II{4-0};
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
}
class Enc_11682941 : OpcodeHexagon {
  bits <19> Ii;
  let Inst{26-25} = Ii{18-17};
  let Inst{20-16} = Ii{16-12};
  let Inst{13-13} = Ii{11-11};
  let Inst{7-0} = Ii{10-3};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
}
class Enc_16376009 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{8-5} = Ii{5-2};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_13249928 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{13-5} = Ii{8-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_1971351 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{8-5} = Ii{4-1};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Ryy32;
  let Inst{4-0} = Ryy32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_12373826 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{13-13} = Ii{10-10};
  let Inst{10-8} = Ii{9-7};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Zdd8;
  let Inst{4-0} = Zdd8{4-0};
}
class Enc_13715847 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{17-16} = Ii{5-4};
  let Inst{6-3} = Ii{3-0};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
}
class Enc_13303422 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{8-5} = Ii{4-1};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_14574598 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{13-8} = Ii{5-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_13094118 : OpcodeHexagon {
  bits <5> Css32;
  let Inst{20-16} = Css32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_4231995 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{13-8} = Ii{5-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_844699 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <4> n1;
  let Inst{28-28} = n1{3-3};
  let Inst{24-22} = n1{2-0};
}
class Enc_8752140 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{8-5} = Ii{5-2};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_7978128 : OpcodeHexagon {
  bits <1> Ii;
  let Inst{8-8} = Ii{0-0};
  bits <2> Qv4;
  let Inst{23-22} = Qv4{1-0};
}
class Enc_10492541 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{6-3} = Ii{5-2};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_0 : OpcodeHexagon {
}
class Enc_8868098 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{21-21} = Ii{9-9};
  let Inst{13-8} = Ii{8-3};
  let Inst{2-0} = Ii{2-0};
  bits <5> Vss32;
  let Inst{7-3} = Vss32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_10380392 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{13-13} = Ii{10-10};
  let Inst{10-8} = Ii{9-7};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
}
class Enc_15733946 : OpcodeHexagon {
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_738356 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{13-13} = Ii{10-10};
  let Inst{10-8} = Ii{9-7};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_15578334 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{10-8} = Ii{9-7};
  bits <5> Zdd8;
  let Inst{4-0} = Zdd8{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_14400220 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{9-5} = Ii{4-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_15194851 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <2> Pu4;
  let Inst{6-5} = Pu4{1-0};
  bits <5> Rx32;
  let Inst{4-0} = Rx32{4-0};
}
class Enc_14172170 : OpcodeHexagon {
  bits <1> Ii;
  let Inst{5-5} = Ii{0-0};
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_10065510 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{6-3} = Ii{5-2};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_14998517 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
  bits <3> n1;
  let Inst{29-29} = n1{2-2};
  let Inst{26-25} = n1{1-0};
}
class Enc_16657398 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{17-16} = Ii{5-4};
  let Inst{6-3} = Ii{3-0};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_14620934 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_10075393 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{13-13} = Ii{9-9};
  let Inst{10-8} = Ii{8-6};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
}
class Enc_8638014 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{21-21} = Ii{15-15};
  let Inst{13-8} = Ii{14-9};
  let Inst{2-0} = Ii{8-6};
  bits <5> Vss32;
  let Inst{7-3} = Vss32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_13261538 : OpcodeHexagon {
  bits <3> Ii;
  let Inst{7-5} = Ii{2-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_8990840 : OpcodeHexagon {
  bits <13> Ii;
  let Inst{26-25} = Ii{12-11};
  let Inst{13-5} = Ii{10-2};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_5974204 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vvv32;
  let Inst{12-8} = Vvv32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_4711514 : OpcodeHexagon {
  bits <2> Qu4;
  let Inst{9-8} = Qu4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_11492529 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{6-3} = Ii{4-1};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_9277990 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_6690615 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{8-4} = Ii{6-2};
  bits <4> Rt16;
  let Inst{3-0} = Rt16{3-0};
}
class Enc_1220199 : OpcodeHexagon {
  bits <2> Qv4;
  let Inst{23-22} = Qv4{1-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_7785569 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <6> n1;
  let Inst{28-28} = n1{5-5};
  let Inst{25-22} = n1{4-1};
  let Inst{8-8} = n1{0-0};
}
class Enc_2880796 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{12-8} = Ii{4-0};
  bits <5> II;
  let Inst{22-21} = II{4-3};
  let Inst{7-5} = II{2-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rx32;
  let Inst{4-0} = Rx32{4-0};
}
class Enc_6858527 : OpcodeHexagon {
  bits <2> Qs4;
  let Inst{6-5} = Qs4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vv32;
  let Inst{4-0} = Vv32{4-0};
}
class Enc_11863656 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rtt32;
  let Inst{20-16} = Rtt32{4-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_151014 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
  bits <2> Px4;
  let Inst{6-5} = Px4{1-0};
}
class Enc_10333841 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{21-21} = Ii{15-15};
  let Inst{13-8} = Ii{14-9};
  let Inst{2-0} = Ii{8-6};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_14044877 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{13-13} = Ii{5-5};
  let Inst{7-3} = Ii{4-0};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_13691337 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
  bits <2> Qx4;
  let Inst{6-5} = Qx4{1-0};
}
class Enc_3817033 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{20-16} = Vuu32{4-0};
  bits <3> Qt8;
  let Inst{10-8} = Qt8{2-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_3540372 : OpcodeHexagon {
  bits <5> Rtt32;
  let Inst{20-16} = Rtt32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_5200852 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_15949334 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_3831744 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_8280533 : OpcodeHexagon {
  bits <3> Ii;
  let Inst{7-5} = Ii{2-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_10969213 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vvv32;
  let Inst{12-8} = Vvv32{4-0};
  bits <5> Vw32;
  let Inst{4-0} = Vw32{4-0};
}
class Enc_3974695 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{10-4} = Ii{6-0};
  bits <4> Rx16;
  let Inst{3-0} = Rx16{3-0};
}
class Enc_7255914 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_7212930 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{8-5} = Ii{4-1};
  bits <2> Pt4;
  let Inst{10-9} = Pt4{1-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_12781442 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <2> Qd4;
  let Inst{1-0} = Qd4{1-0};
}
class Enc_799555 : OpcodeHexagon {
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_11083408 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{23-19} = Vv32{4-0};
  bits <3> Rt8;
  let Inst{18-16} = Rt8{2-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_900013 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_9487067 : OpcodeHexagon {
  bits <12> Ii;
  let Inst{19-16} = Ii{11-8};
  let Inst{12-5} = Ii{7-0};
  bits <2> Pu4;
  let Inst{22-21} = Pu4{1-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_16014536 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{21-21} = Ii{9-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_12419313 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <4> n1;
  let Inst{28-28} = n1{3-3};
  let Inst{24-23} = n1{2-1};
  let Inst{13-13} = n1{0-0};
}
class Enc_5503430 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_14767681 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{23-19} = Vv32{4-0};
  bits <3> Rt8;
  let Inst{18-16} = Rt8{2-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_9093094 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-5} = Ii{7-0};
  bits <8> II;
  let Inst{22-16} = II{7-1};
  let Inst{13-13} = II{0-0};
  bits <2> Pu4;
  let Inst{24-23} = Pu4{1-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_11542684 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{27-21} = Ii{15-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_8877260 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{23-19} = Vv32{4-0};
  bits <3> Rt8;
  let Inst{18-16} = Rt8{2-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_1737833 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{13-13} = Ii{5-5};
  let Inst{7-3} = Ii{4-0};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_255516 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{20-16} = Vuu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_10721363 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_7076358 : OpcodeHexagon {
  bits <5> Zdd8;
  let Inst{4-0} = Zdd8{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_11930928 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{12-8} = Ii{4-0};
  bits <5> II;
  let Inst{22-21} = II{4-3};
  let Inst{7-5} = II{2-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_2410156 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{12-8} = Ii{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rx32;
  let Inst{4-0} = Rx32{4-0};
}
class Enc_6735062 : OpcodeHexagon {
  bits <2> Ps4;
  let Inst{17-16} = Ps4{1-0};
  bits <2> Pt4;
  let Inst{9-8} = Pt4{1-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_7965855 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_5202340 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vyy32;
  let Inst{4-0} = Vyy32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_10568534 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-5} = Ii{7-0};
  bits <2> Pu4;
  let Inst{22-21} = Pu4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_16730127 : OpcodeHexagon {
  bits <3> Ii;
  let Inst{7-5} = Ii{2-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_11224149 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{13-13} = Ii{7-7};
  let Inst{7-3} = Ii{6-2};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_9772987 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ru32;
  let Inst{12-8} = Ru32{4-0};
  bits <5> Rtt32;
  let Inst{4-0} = Rtt32{4-0};
}
class Enc_9238139 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Zdd8;
  let Inst{4-0} = Zdd8{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_2082775 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{11-8} = Ii{3-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_5790679 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{12-8} = Ii{8-4};
  let Inst{4-3} = Ii{3-2};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_9305257 : OpcodeHexagon {
  bits <5> Zu8;
  let Inst{12-8} = Zu8{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_3735566 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-8} = Ii{8-6};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_12654528 : OpcodeHexagon {
  bits <2> Qs4;
  let Inst{6-5} = Qs4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vvv32;
  let Inst{4-0} = Vvv32{4-0};
}
class Enc_15290236 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_11139981 : OpcodeHexagon {
  bits <2> Ps4;
  let Inst{17-16} = Ps4{1-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_15546666 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-8} = Ii{8-6};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_486163 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <6> II;
  let Inst{11-8} = II{5-2};
  let Inst{6-5} = II{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_2079016 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{1-0} = Ii{1-0};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
}
class Enc_10095813 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rtt32;
  let Inst{20-16} = Rtt32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_13133322 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vx32;
  let Inst{7-3} = Vx32{4-0};
}
class Enc_9422954 : OpcodeHexagon {
  bits <2> Pu4;
  let Inst{9-8} = Pu4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_10642833 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vs32;
  let Inst{7-3} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_14989332 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vv32;
  let Inst{4-0} = Vv32{4-0};
}
class Enc_10263630 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <3> Rt8;
  let Inst{2-0} = Rt8{2-0};
  bits <5> Vx32;
  let Inst{7-3} = Vx32{4-0};
}
class Enc_13937564 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{13-13} = Ii{10-10};
  let Inst{10-8} = Ii{9-7};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
}
class Enc_7171569 : OpcodeHexagon {
  bits <3> Ii;
  let Inst{7-5} = Ii{2-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_2702036 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{21-21} = Ii{9-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_1928953 : OpcodeHexagon {
  bits <2> Pu4;
  let Inst{9-8} = Pu4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_5853469 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <2> Pe4;
  let Inst{6-5} = Pe4{1-0};
}
class Enc_7692963 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rx32;
  let Inst{4-0} = Rx32{4-0};
}
class Enc_15140689 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_748676 : OpcodeHexagon {
  bits <12> Ii;
  let Inst{26-25} = Ii{11-10};
  let Inst{13-13} = Ii{9-9};
  let Inst{7-0} = Ii{8-1};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_3372766 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{8-5} = Ii{4-1};
  bits <5> Ryy32;
  let Inst{4-0} = Ryy32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_7900405 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{6-3} = Ii{5-2};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_11930027 : OpcodeHexagon {
  bits <12> Ii;
  let Inst{26-25} = Ii{11-10};
  let Inst{13-5} = Ii{9-1};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ryy32;
  let Inst{4-0} = Ryy32{4-0};
}
class Enc_971574 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{22-21} = Ii{5-4};
  let Inst{13-13} = Ii{3-3};
  let Inst{7-5} = Ii{2-0};
  bits <6> II;
  let Inst{23-23} = II{5-5};
  let Inst{4-0} = II{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{12-8} = Rd32{4-0};
}
class Enc_13453446 : OpcodeHexagon {
  bits <24> Ii;
  let Inst{24-16} = Ii{23-15};
  let Inst{13-1} = Ii{14-2};
}
class Enc_6356866 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{21-21} = Ii{9-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rx32;
  let Inst{4-0} = Rx32{4-0};
}
class Enc_16246706 : OpcodeHexagon {
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_5326450 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{6-3} = Ii{3-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_11687333 : OpcodeHexagon {
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_2771456 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{12-8} = Ii{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_11282123 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{12-7} = Ii{5-0};
  bits <8> II;
  let Inst{13-13} = II{7-7};
  let Inst{6-0} = II{6-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_518319 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{20-16} = Ii{5-1};
  let Inst{5-5} = Ii{0-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_16104442 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rtt32;
  let Inst{20-16} = Rtt32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_7912540 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rxx32;
  let Inst{4-0} = Rxx32{4-0};
}
class Enc_15560488 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{10-8} = Ii{9-7};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_7581852 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_10030031 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_3915770 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{6-3} = Ii{3-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_4075554 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_11326438 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{6-3} = Ii{5-2};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_4050532 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{26-25} = Ii{15-14};
  let Inst{20-16} = Ii{13-9};
  let Inst{13-13} = Ii{8-8};
  let Inst{7-0} = Ii{7-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_14461004 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{26-25} = Ii{10-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_13344657 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{20-16} = Ii{5-1};
  let Inst{8-8} = Ii{0-0};
  bits <2> Pt4;
  let Inst{10-9} = Pt4{1-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_13114546 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{5-5} = Ii{0-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rxx32;
  let Inst{4-0} = Rxx32{4-0};
}
class Enc_14530015 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <6> n1;
  let Inst{28-28} = n1{5-5};
  let Inst{25-23} = n1{4-2};
  let Inst{13-13} = n1{1-1};
  let Inst{8-8} = n1{0-0};
}
class Enc_5967898 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{12-7} = Ii{5-0};
  bits <6> II;
  let Inst{13-13} = II{5-5};
  let Inst{4-0} = II{4-0};
  bits <2> Pv4;
  let Inst{6-5} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_15450971 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <6> n1;
  let Inst{28-28} = n1{5-5};
  let Inst{25-22} = n1{4-1};
  let Inst{13-13} = n1{0-0};
}
class Enc_15536400 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{3-0} = Ii{5-2};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
}
class Enc_1291652 : OpcodeHexagon {
  bits <1> Ii;
  let Inst{8-8} = Ii{0-0};
}
class Enc_5636753 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
}
class Enc_5757366 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{13-13} = Ii{10-10};
  let Inst{10-8} = Ii{9-7};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
}
class Enc_9752128 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{8-5} = Ii{6-3};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_13618890 : OpcodeHexagon {
  bits <17> Ii;
  let Inst{26-25} = Ii{16-15};
  let Inst{20-16} = Ii{14-10};
  let Inst{13-13} = Ii{9-9};
  let Inst{7-0} = Ii{8-1};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_5890213 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_5582416 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <6> II;
  let Inst{11-8} = II{5-2};
  let Inst{6-5} = II{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_13536408 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{3-0} = Ii{3-0};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
}
class Enc_9773189 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Ru32;
  let Inst{4-0} = Ru32{4-0};
  bits <5> Rxx32;
  let Inst{12-8} = Rxx32{4-0};
}
class Enc_2152247 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{13-13} = Ii{10-10};
  let Inst{10-8} = Ii{9-7};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
}
class Enc_12848507 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{6-6} = Ii{0-0};
  bits <6> II;
  let Inst{5-0} = II{5-0};
  bits <5> Ru32;
  let Inst{20-16} = Ru32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
}
class Enc_16279406 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{13-13} = Ii{9-9};
  let Inst{10-8} = Ii{8-6};
  bits <2> Qv4;
  let Inst{12-11} = Qv4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
}
class Enc_1734121 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{10-8} = Ii{3-1};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rt16;
  let Inst{3-0} = Rt16{3-0};
}
class Enc_766909 : OpcodeHexagon {
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
  bits <2> Pe4;
  let Inst{6-5} = Pe4{1-0};
}
class Enc_4527648 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_8849208 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{12-7} = Ii{6-1};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{4-0} = Rt32{4-0};
}
class Enc_9894557 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{13-8} = Ii{5-0};
  bits <6> II;
  let Inst{23-21} = II{5-3};
  let Inst{7-5} = II{2-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_4109168 : OpcodeHexagon {
  bits <2> Qv4;
  let Inst{23-22} = Qv4{1-0};
}
class Enc_14560494 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-8} = Ii{8-6};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_9773167 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{12-7} = Ii{6-1};
  bits <5> II;
  let Inst{4-0} = II{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_1898420 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
}
class Enc_11498120 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <2> Qd4;
  let Inst{1-0} = Qd4{1-0};
}
class Enc_15459921 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-8} = Ii{8-6};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_10058269 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_10197700 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{20-16} = Vuu32{4-0};
  bits <5> Vvv32;
  let Inst{12-8} = Vvv32{4-0};
  bits <3> Rt8;
  let Inst{2-0} = Rt8{2-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_12608570 : OpcodeHexagon {
  bits <17> Ii;
  let Inst{26-25} = Ii{16-15};
  let Inst{20-16} = Ii{14-10};
  let Inst{13-5} = Ii{9-1};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_4804090 : OpcodeHexagon {
  bits <6> Ss64;
  let Inst{21-16} = Ss64{5-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_14973146 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <3> Qd8;
  let Inst{5-3} = Qd8{2-0};
}
class Enc_5718302 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <2> Pe4;
  let Inst{6-5} = Pe4{1-0};
}
class Enc_2103742 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{12-8} = Ii{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_7564330 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <3> Rt8;
  let Inst{2-0} = Rt8{2-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_2176383 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{9-4} = Ii{5-0};
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
}
class Enc_7736768 : OpcodeHexagon {
  bits <12> Ii;
  let Inst{26-25} = Ii{11-10};
  let Inst{13-13} = Ii{9-9};
  let Inst{7-0} = Ii{8-1};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_13189194 : OpcodeHexagon {
  bits <1> Ii;
  let Inst{5-5} = Ii{0-0};
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vxx32;
  let Inst{4-0} = Vxx32{4-0};
}
class Enc_5154851 : OpcodeHexagon {
  bits <5> Rtt32;
  let Inst{20-16} = Rtt32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_1329520 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Cdd32;
  let Inst{4-0} = Cdd32{4-0};
}
class Enc_14057553 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{21-21} = Ii{15-15};
  let Inst{13-8} = Ii{14-9};
  let Inst{2-0} = Ii{8-6};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_9223889 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{4-0} = Rx32{4-0};
}
class Enc_10979813 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{13-13} = Ii{6-6};
  let Inst{7-3} = Ii{5-1};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_13490067 : OpcodeHexagon {
  bits <3> Qt8;
  let Inst{2-0} = Qt8{2-0};
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_10076500 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{6-6} = Ii{0-0};
  bits <6> II;
  let Inst{5-0} = II{5-0};
  bits <5> Ru32;
  let Inst{20-16} = Ru32{4-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_163381 : OpcodeHexagon {
  bits <14> Ii;
  let Inst{26-25} = Ii{13-12};
  let Inst{13-5} = Ii{11-3};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_10328975 : OpcodeHexagon {
  bits <2> Pt4;
  let Inst{9-8} = Pt4{1-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_14939491 : OpcodeHexagon {
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
}
class Enc_8891794 : OpcodeHexagon {
  bits <2> Pt4;
  let Inst{9-8} = Pt4{1-0};
  bits <2> Ps4;
  let Inst{17-16} = Ps4{1-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_7723767 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_2639299 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <4> Rd16;
  let Inst{11-8} = Rd16{3-0};
}
class Enc_11552785 : OpcodeHexagon {
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <2> Pu4;
  let Inst{6-5} = Pu4{1-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_11849200 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{12-7} = Ii{5-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{4-0} = Rt32{4-0};
}
class Enc_14868535 : OpcodeHexagon {
  bits <17> Ii;
  let Inst{23-22} = Ii{16-15};
  let Inst{20-16} = Ii{14-10};
  let Inst{13-13} = Ii{9-9};
  let Inst{7-1} = Ii{8-2};
  bits <2> Pu4;
  let Inst{9-8} = Pu4{1-0};
}
class Enc_48594 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_6608821 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{13-13} = Ii{9-9};
  let Inst{10-8} = Ii{8-6};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
}
class Enc_11049656 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{13-13} = Ii{8-8};
  let Inst{7-3} = Ii{7-3};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
}
class Enc_117962 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{23-21} = Ii{7-5};
  let Inst{13-13} = Ii{4-4};
  let Inst{7-5} = Ii{3-1};
  let Inst{3-3} = Ii{0-0};
  bits <5> II;
  let Inst{12-8} = II{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_5900401 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{6-3} = Ii{3-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_36641 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_9626139 : OpcodeHexagon {
  bits <2> Pu4;
  let Inst{6-5} = Pu4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_11971407 : OpcodeHexagon {
  bits <3> Ii;
  let Inst{7-5} = Ii{2-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_9852473 : OpcodeHexagon {
  bits <13> Ii;
  let Inst{26-25} = Ii{12-11};
  let Inst{13-5} = Ii{10-2};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_6495334 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{22-21} = Ii{5-4};
  let Inst{13-13} = Ii{3-3};
  let Inst{7-5} = Ii{2-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ru32;
  let Inst{4-0} = Ru32{4-0};
  bits <5> Rd32;
  let Inst{12-8} = Rd32{4-0};
}
class Enc_1186018 : OpcodeHexagon {
  bits <17> Ii;
  let Inst{26-25} = Ii{16-15};
  let Inst{20-16} = Ii{14-10};
  let Inst{13-13} = Ii{9-9};
  let Inst{7-0} = Ii{8-1};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_15999208 : OpcodeHexagon {
  bits <18> Ii;
  let Inst{26-25} = Ii{17-16};
  let Inst{20-16} = Ii{15-11};
  let Inst{13-13} = Ii{10-10};
  let Inst{7-0} = Ii{9-2};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_11477246 : OpcodeHexagon {
  bits <6> II;
  let Inst{5-0} = II{5-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Re32;
  let Inst{20-16} = Re32{4-0};
}
class Enc_7971062 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{23-22} = Ii{15-14};
  let Inst{20-16} = Ii{13-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_4327792 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vxx32;
  let Inst{4-0} = Vxx32{4-0};
}
class Enc_10326434 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{6-3} = Ii{4-1};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_1572239 : OpcodeHexagon {
  bits <2> Qt4;
  let Inst{6-5} = Qt4{1-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_6372758 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{8-5} = Ii{3-0};
  bits <5> Ryy32;
  let Inst{4-0} = Ryy32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_15793331 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vx32;
  let Inst{7-3} = Vx32{4-0};
}
class Enc_11424254 : OpcodeHexagon {
  bits <2> Qt4;
  let Inst{6-5} = Qt4{1-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_4983213 : OpcodeHexagon {
  bits <14> Ii;
  let Inst{10-0} = Ii{13-3};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_16035138 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
}
class Enc_8225953 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{13-13} = Ii{7-7};
  let Inst{7-3} = Ii{6-2};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_4397470 : OpcodeHexagon {
  bits <5> II;
  let Inst{12-8} = II{4-0};
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
}
class Enc_1004392 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vxx32;
  let Inst{7-3} = Vxx32{4-0};
}
class Enc_16319737 : OpcodeHexagon {
  bits <14> Ii;
  let Inst{26-25} = Ii{13-12};
  let Inst{13-13} = Ii{11-11};
  let Inst{7-0} = Ii{10-3};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
}
class Enc_2296022 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{10-8} = Ii{9-7};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_14546668 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{10-8} = Ii{9-7};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_9664427 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{20-16} = Vuu32{4-0};
  bits <5> Vvv32;
  let Inst{12-8} = Vvv32{4-0};
  bits <3> Qss8;
  let Inst{2-0} = Qss8{2-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_877823 : OpcodeHexagon {
  bits <6> II;
  let Inst{11-8} = II{5-2};
  let Inst{6-5} = II{1-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
  bits <5> Re32;
  let Inst{20-16} = Re32{4-0};
}
class Enc_1589406 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_6900405 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{6-3} = Ii{4-1};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_14150875 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <5> n1;
  let Inst{28-28} = n1{4-4};
  let Inst{25-22} = n1{3-0};
}
class Enc_15707793 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Gd32;
  let Inst{4-0} = Gd32{4-0};
}
class Enc_14689096 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{6-6} = Ii{0-0};
  bits <6> II;
  let Inst{5-0} = II{5-0};
  bits <5> Ru32;
  let Inst{20-16} = Ru32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_9915754 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{6-3} = Ii{5-2};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_7470998 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <2> Qx4;
  let Inst{1-0} = Qx4{1-0};
}
class Enc_11471622 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_14363183 : OpcodeHexagon {
  bits <2> Qv4;
  let Inst{23-22} = Qv4{1-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_15816255 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_5321335 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <3> Rt8;
  let Inst{2-0} = Rt8{2-0};
  bits <4> Vdd16;
  let Inst{7-4} = Vdd16{3-0};
}
class Enc_12702821 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rxx32;
  let Inst{4-0} = Rxx32{4-0};
}
class Enc_449439 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{26-25} = Ii{10-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ryy32;
  let Inst{4-0} = Ryy32{4-0};
}
class Enc_2054304 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <6> Sd64;
  let Inst{5-0} = Sd64{5-0};
}
class Enc_236434 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{22-21} = Ii{5-4};
  let Inst{13-13} = Ii{3-3};
  let Inst{7-5} = Ii{2-0};
  bits <5> Ru32;
  let Inst{4-0} = Ru32{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{12-8} = Rd32{4-0};
}
class Enc_5598813 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{8-5} = Ii{3-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_8409782 : OpcodeHexagon {
  bits <13> Ii;
  let Inst{26-25} = Ii{12-11};
  let Inst{13-13} = Ii{10-10};
  let Inst{7-0} = Ii{9-2};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_15182416 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{20-16} = Ii{5-1};
  let Inst{8-8} = Ii{0-0};
  bits <2> Pt4;
  let Inst{10-9} = Pt4{1-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_4501395 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{6-3} = Ii{6-3};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_6039436 : OpcodeHexagon {
  bits <3> Qtt8;
  let Inst{2-0} = Qtt8{2-0};
  bits <5> Vuu32;
  let Inst{20-16} = Vuu32{4-0};
  bits <5> Vvv32;
  let Inst{12-8} = Vvv32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_476163 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <3> Rt8;
  let Inst{2-0} = Rt8{2-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
  bits <5> Vy32;
  let Inst{12-8} = Vy32{4-0};
}
class Enc_11281763 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_9929262 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{21-21} = Ii{15-15};
  let Inst{13-8} = Ii{14-9};
  let Inst{2-0} = Ii{8-6};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vs32;
  let Inst{7-3} = Vs32{4-0};
}
class Enc_13174858 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{21-21} = Ii{15-15};
  let Inst{13-8} = Ii{14-9};
  let Inst{2-0} = Ii{8-6};
  bits <5> Vs32;
  let Inst{7-3} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_8437395 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{13-13} = Ii{10-10};
  let Inst{10-8} = Ii{9-7};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_16578332 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-8} = Ii{8-6};
  bits <5> Zdd8;
  let Inst{4-0} = Zdd8{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_12829314 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
}
class Enc_9744403 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{13-9} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{8-4} = Vv32{4-0};
  bits <4> Vdd16;
  let Inst{3-0} = Vdd16{3-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_10968391 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <7> n1;
  let Inst{28-28} = n1{6-6};
  let Inst{25-22} = n1{5-2};
  let Inst{13-13} = n1{1-1};
  let Inst{8-8} = n1{0-0};
}
class Enc_64199 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{8-4} = Ii{6-2};
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
}
class Enc_11039423 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{10-8} = Ii{9-7};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_6730375 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
}
class Enc_16213761 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{23-19} = Vv32{4-0};
  bits <3> Rt8;
  let Inst{18-16} = Rt8{2-0};
  bits <5> Vxx32;
  let Inst{4-0} = Vxx32{4-0};
}
class Enc_13204995 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{11-8} = Ii{3-0};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rt16;
  let Inst{3-0} = Rt16{3-0};
}
class Enc_13338314 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{13-13} = Ii{9-9};
  let Inst{10-8} = Ii{8-6};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_9920336 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <2> Pv4;
  let Inst{6-5} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ru32;
  let Inst{12-8} = Ru32{4-0};
  bits <5> Rtt32;
  let Inst{4-0} = Rtt32{4-0};
}
class Enc_15380240 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <3> Rt8;
  let Inst{2-0} = Rt8{2-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
  bits <5> Vy32;
  let Inst{12-8} = Vy32{4-0};
}
class Enc_3296020 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-8} = Ii{8-6};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_2428539 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <4> n1;
  let Inst{28-28} = n1{3-3};
  let Inst{24-23} = n1{2-1};
  let Inst{8-8} = n1{0-0};
}
class Enc_10039393 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-8} = Ii{8-6};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_9372046 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{13-13} = Ii{9-9};
  let Inst{10-8} = Ii{8-6};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
}
class Enc_2901241 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_16145290 : OpcodeHexagon {
  bits <2> Ps4;
  let Inst{6-5} = Ps4{1-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_5555790 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{21-21} = Ii{9-9};
  let Inst{13-8} = Ii{8-3};
  let Inst{2-0} = Ii{2-0};
  bits <5> Vs32;
  let Inst{7-3} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_13783220 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rtt32;
  let Inst{20-16} = Rtt32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_12261611 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Ryy32;
  let Inst{4-0} = Ryy32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_6135183 : OpcodeHexagon {
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rx16;
  let Inst{3-0} = Rx16{3-0};
}
class Enc_5523416 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{13-8} = Ii{5-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_13472494 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{21-21} = Ii{9-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_16303398 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{8-5} = Ii{3-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_3494181 : OpcodeHexagon {
  bits <3> Ii;
  let Inst{7-5} = Ii{2-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_13983714 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <2> Qd4;
  let Inst{1-0} = Qd4{1-0};
}
class Enc_931653 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{8-5} = Ii{6-3};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_7622936 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <3> Rt8;
  let Inst{2-0} = Rt8{2-0};
  bits <5> Vxx32;
  let Inst{7-3} = Vxx32{4-0};
  bits <5> Vy32;
  let Inst{12-8} = Vy32{4-0};
}
class Enc_8773155 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-7} = Ii{7-2};
  bits <5> II;
  let Inst{4-0} = II{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_5401217 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <3> n1;
  let Inst{28-28} = n1{2-2};
  let Inst{24-23} = n1{1-0};
}
class Enc_6736678 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-5} = Ii{7-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_3457570 : OpcodeHexagon {
  bits <3> Ii;
  let Inst{7-5} = Ii{2-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vxx32;
  let Inst{4-0} = Vxx32{4-0};
}
class Enc_3813442 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{6-3} = Ii{4-1};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_3135259 : OpcodeHexagon {
  bits <3> Ii;
  let Inst{10-8} = Ii{2-0};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
}
class Enc_5486172 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ru32;
  let Inst{12-8} = Ru32{4-0};
  bits <3> Nt8;
  let Inst{2-0} = Nt8{2-0};
}
class Enc_11081334 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{21-21} = Ii{15-15};
  let Inst{13-8} = Ii{14-9};
  let Inst{2-0} = Ii{8-6};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vss32;
  let Inst{7-3} = Vss32{4-0};
}
class Enc_9470751 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{13-13} = Ii{10-10};
  let Inst{10-8} = Ii{9-7};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
}
class Enc_2683366 : OpcodeHexagon {
  bits <3> Quu8;
  let Inst{10-8} = Quu8{2-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <3> Qdd8;
  let Inst{5-3} = Qdd8{2-0};
}
class Enc_15830826 : OpcodeHexagon {
  bits <14> Ii;
  let Inst{10-0} = Ii{13-3};
}
class Enc_4967902 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{12-7} = Ii{6-1};
  bits <6> II;
  let Inst{13-13} = II{5-5};
  let Inst{4-0} = II{4-0};
  bits <2> Pv4;
  let Inst{6-5} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_14287645 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_8324216 : OpcodeHexagon {
  bits <2> Ps4;
  let Inst{17-16} = Ps4{1-0};
  bits <2> Pt4;
  let Inst{9-8} = Pt4{1-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_913538 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <3> Qd8;
  let Inst{5-3} = Qd8{2-0};
}
class Enc_16311032 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rx32;
  let Inst{4-0} = Rx32{4-0};
}
class Enc_9864697 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-5} = Ii{7-0};
  bits <6> II;
  let Inst{20-16} = II{5-1};
  let Inst{13-13} = II{0-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_11205051 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{11-8} = Ii{5-2};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rt16;
  let Inst{3-0} = Rt16{3-0};
}
class Enc_5611087 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{8-5} = Ii{6-3};
  bits <2> Pt4;
  let Inst{10-9} = Pt4{1-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_10915758 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{6-3} = Ii{4-1};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_8943121 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
}
class Enc_1539665 : OpcodeHexagon {
  bits <5> Cs32;
  let Inst{20-16} = Cs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_8479583 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
  bits <5> n1;
  let Inst{29-29} = n1{4-4};
  let Inst{26-25} = n1{3-2};
  let Inst{23-23} = n1{1-1};
  let Inst{13-13} = n1{0-0};
}
class Enc_313333 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_11544269 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
  bits <4> n1;
  let Inst{29-29} = n1{3-3};
  let Inst{26-25} = n1{2-1};
  let Inst{13-13} = n1{0-0};
}
class Enc_9018141 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Cd32;
  let Inst{4-0} = Cd32{4-0};
}
class Enc_6152036 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Gdd32;
  let Inst{4-0} = Gdd32{4-0};
}
class Enc_1954437 : OpcodeHexagon {
  bits <6> Sss64;
  let Inst{21-16} = Sss64{5-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_3742184 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_1835415 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{10-5} = Ii{6-1};
  bits <2> Pt4;
  let Inst{12-11} = Pt4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_1085466 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_13150110 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{26-25} = Ii{10-9};
  let Inst{13-13} = Ii{8-8};
  let Inst{7-0} = Ii{7-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_6772177 : OpcodeHexagon {
  bits <5> Zu8;
  let Inst{12-8} = Zu8{4-0};
  bits <5> Zd8;
  let Inst{4-0} = Zd8{4-0};
}
class Enc_6616512 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{21-21} = Ii{15-15};
  let Inst{13-8} = Ii{14-9};
  let Inst{2-0} = Ii{8-6};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_1886960 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{26-25} = Ii{15-14};
  let Inst{20-16} = Ii{13-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_2835415 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{10-5} = Ii{7-2};
  bits <2> Pt4;
  let Inst{12-11} = Pt4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_14024197 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rtt32;
  let Inst{20-16} = Rtt32{4-0};
  bits <5> Vxx32;
  let Inst{4-0} = Vxx32{4-0};
}
class Enc_12297800 : OpcodeHexagon {
  bits <18> Ii;
  let Inst{26-25} = Ii{17-16};
  let Inst{20-16} = Ii{15-11};
  let Inst{13-13} = Ii{10-10};
  let Inst{7-0} = Ii{9-2};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_7254313 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <2> Pv4;
  let Inst{6-5} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_677558 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-5} = Ii{8-3};
  bits <2> Pt4;
  let Inst{12-11} = Pt4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_6223403 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_674613 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{20-16} = Vuu32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_16479122 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{7-3} = Ii{7-3};
  bits <3> Rdd8;
  let Inst{2-0} = Rdd8{2-0};
}
class Enc_11704059 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_9165078 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{8-3} = Ii{8-3};
  bits <3> Rtt8;
  let Inst{2-0} = Rtt8{2-0};
}
class Enc_15376009 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{8-5} = Ii{4-1};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_8838398 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{21-21} = Ii{3-3};
  let Inst{7-5} = Ii{2-0};
  bits <6> II;
  let Inst{13-8} = II{5-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rx32;
  let Inst{4-0} = Rx32{4-0};
}
class Enc_2328527 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_1451363 : OpcodeHexagon {
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
}
class Enc_4030179 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_13770697 : OpcodeHexagon {
  bits <5> Ru32;
  let Inst{4-0} = Ru32{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ry32;
  let Inst{12-8} = Ry32{4-0};
}
class Enc_12212978 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{8-5} = Ii{3-0};
  bits <2> Pt4;
  let Inst{10-9} = Pt4{1-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_12665927 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_2082956 : OpcodeHexagon {
  bits <32> Ii;
  let Inst{27-16} = Ii{31-20};
  let Inst{13-0} = Ii{19-6};
}
class Enc_220949 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <5> n1;
  let Inst{28-28} = n1{4-4};
  let Inst{25-23} = n1{3-1};
  let Inst{13-13} = n1{0-0};
}
class Enc_9939385 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{12-8} = Ii{8-4};
  let Inst{4-3} = Ii{3-2};
  bits <10> II;
  let Inst{20-16} = II{9-5};
  let Inst{7-5} = II{4-2};
  let Inst{1-0} = II{1-0};
}
class Enc_2117024 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-8} = Ii{7-3};
  let Inst{4-2} = Ii{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_8390029 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{20-16} = Vuu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_10989558 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_5972412 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{20-16} = Vv32{4-0};
  bits <5> Vxx32;
  let Inst{4-0} = Vxx32{4-0};
}
class Enc_12851489 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vss32;
  let Inst{7-3} = Vss32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_9554661 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{12-7} = Ii{5-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_4202401 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_6091631 : OpcodeHexagon {
  bits <2> Qs4;
  let Inst{9-8} = Qs4{1-0};
  bits <2> Qt4;
  let Inst{23-22} = Qt4{1-0};
  bits <2> Qd4;
  let Inst{1-0} = Qd4{1-0};
}
class Enc_10157519 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_4835423 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{10-5} = Ii{5-0};
  bits <2> Pt4;
  let Inst{12-11} = Pt4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_14046916 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ru32;
  let Inst{12-8} = Ru32{4-0};
  bits <5> Rt32;
  let Inst{4-0} = Rt32{4-0};
}
class Enc_2921694 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_8732960 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-8} = Ii{7-3};
  let Inst{4-2} = Ii{2-0};
}
class Enc_5338033 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <5> n1;
  let Inst{28-28} = n1{4-4};
  let Inst{24-22} = n1{3-1};
  let Inst{13-13} = n1{0-0};
}
class Enc_6956613 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_2153798 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vxx32;
  let Inst{4-0} = Vxx32{4-0};
}
class Enc_16210172 : OpcodeHexagon {
  bits <3> Qt8;
  let Inst{10-8} = Qt8{2-0};
  bits <3> Qd8;
  let Inst{5-3} = Qd8{2-0};
}
class Enc_5023792 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_1244745 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{13-13} = Ii{9-9};
  let Inst{10-8} = Ii{8-6};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_10002182 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{26-25} = Ii{10-9};
  let Inst{13-13} = Ii{8-8};
  let Inst{7-0} = Ii{7-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_12492533 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{6-3} = Ii{3-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_1774350 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{17-16} = Ii{5-4};
  let Inst{6-3} = Ii{3-0};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_2703240 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{13-13} = Ii{10-10};
  let Inst{10-8} = Ii{9-7};
  bits <2> Qv4;
  let Inst{12-11} = Qv4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
}
class Enc_6975103 : OpcodeHexagon {
  bits <2> Ps4;
  let Inst{17-16} = Ps4{1-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_9789480 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_12244921 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-8} = Ii{8-6};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_8674673 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
  bits <5> n1;
  let Inst{29-29} = n1{4-4};
  let Inst{26-25} = n1{3-2};
  let Inst{23-22} = n1{1-0};
}
class Enc_8514936 : OpcodeHexagon {
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_13455308 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-5} = Ii{7-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_10188026 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{13-8} = Ii{5-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_3158657 : OpcodeHexagon {
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_10597934 : OpcodeHexagon {
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
  bits <2> n1;
  let Inst{9-8} = n1{1-0};
}
class Enc_10612292 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <2> Qx4;
  let Inst{1-0} = Qx4{1-0};
}
class Enc_5178985 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <2> Pu4;
  let Inst{6-5} = Pu4{1-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_3967902 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-7} = Ii{7-2};
  bits <6> II;
  let Inst{13-13} = II{5-5};
  let Inst{4-0} = II{4-0};
  bits <2> Pv4;
  let Inst{6-5} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_2462143 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-5} = Ii{7-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_9849208 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-7} = Ii{7-2};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{4-0} = Rt32{4-0};
}
class Enc_12618352 : OpcodeHexagon {
  bits <5> Rtt32;
  let Inst{20-16} = Rtt32{4-0};
  bits <5> Vx32;
  let Inst{7-3} = Vx32{4-0};
}
class Enc_7303598 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <6> II;
  let Inst{11-8} = II{5-2};
  let Inst{6-5} = II{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Ryy32;
  let Inst{4-0} = Ryy32{4-0};
}
class Enc_13823098 : OpcodeHexagon {
  bits <5> Gss32;
  let Inst{20-16} = Gss32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_16388420 : OpcodeHexagon {
  bits <2> Qs4;
  let Inst{6-5} = Qs4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vvv32;
  let Inst{12-8} = Vvv32{4-0};
  bits <5> Vw32;
  let Inst{4-0} = Vw32{4-0};
}
class Enc_8328140 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{21-21} = Ii{15-15};
  let Inst{13-8} = Ii{14-9};
  let Inst{2-0} = Ii{8-6};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_1793896 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <2> Pv4;
  let Inst{6-5} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_4944558 : OpcodeHexagon {
  bits <2> Qu4;
  let Inst{9-8} = Qu4{1-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vx32;
  let Inst{4-0} = Vx32{4-0};
}
class Enc_13211717 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Vvv32;
  let Inst{20-16} = Vvv32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_8170340 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vx32;
  let Inst{7-3} = Vx32{4-0};
  bits <3> Qdd8;
  let Inst{2-0} = Qdd8{2-0};
}
class Enc_14071773 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_8605375 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_12711252 : OpcodeHexagon {
  bits <2> Pv4;
  let Inst{9-8} = Pv4{1-0};
}
class Enc_8202458 : OpcodeHexagon {
  bits <2> Pu4;
  let Inst{6-5} = Pu4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_8577055 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <5> n1;
  let Inst{28-28} = n1{4-4};
  let Inst{25-23} = n1{3-1};
  let Inst{8-8} = n1{0-0};
}
class Enc_1409050 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rxx32;
  let Inst{4-0} = Rxx32{4-0};
}
class Enc_7466005 : OpcodeHexagon {
  bits <5> Gs32;
  let Inst{20-16} = Gs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_2380082 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{12-8} = Ii{4-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_10067774 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_11000933 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <2> Pv4;
  let Inst{6-5} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ru32;
  let Inst{12-8} = Ru32{4-0};
  bits <3> Nt8;
  let Inst{2-0} = Nt8{2-0};
}
class Enc_13201267 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{12-8} = Ii{4-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_1989309 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vvv32;
  let Inst{4-0} = Vvv32{4-0};
}
class Enc_9082775 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{21-21} = Ii{9-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_8065534 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{6-3} = Ii{3-0};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_4631106 : OpcodeHexagon {
  bits <2> Ps4;
  let Inst{17-16} = Ps4{1-0};
  bits <2> Pt4;
  let Inst{9-8} = Pt4{1-0};
  bits <2> Pu4;
  let Inst{7-6} = Pu4{1-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_11065510 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{6-3} = Ii{4-1};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_8829170 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{13-13} = Ii{9-9};
  let Inst{10-8} = Ii{8-6};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
}
class Enc_6673186 : OpcodeHexagon {
  bits <13> Ii;
  let Inst{26-25} = Ii{12-11};
  let Inst{13-13} = Ii{10-10};
  let Inst{7-0} = Ii{9-2};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_8498433 : OpcodeHexagon {
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_4395009 : OpcodeHexagon {
  bits <7> Ii;
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_10926598 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{12-8} = Vuu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vxx32;
  let Inst{7-3} = Vxx32{4-0};
}
class Enc_7606379 : OpcodeHexagon {
  bits <2> Pu4;
  let Inst{6-5} = Pu4{1-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_8131399 : OpcodeHexagon {
  bits <6> II;
  let Inst{5-0} = II{5-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Re32;
  let Inst{20-16} = Re32{4-0};
}
class Enc_11522288 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-5} = Ii{7-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rx32;
  let Inst{4-0} = Rx32{4-0};
}
class Enc_114098 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{5-5} = Ii{0-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_5654851 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{12-8} = Ii{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_12023037 : OpcodeHexagon {
  bits <2> Ps4;
  let Inst{6-5} = Ps4{1-0};
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_176263 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{9-4} = Ii{7-2};
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
}
class Enc_6130414 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{23-22} = Ii{15-14};
  let Inst{13-0} = Ii{13-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_631197 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{13-8} = Ii{5-0};
  bits <6> II;
  let Inst{23-21} = II{5-3};
  let Inst{7-5} = II{2-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rxx32;
  let Inst{4-0} = Rxx32{4-0};
}
class Enc_16214129 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_8333157 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_4834775 : OpcodeHexagon {
  bits <6> II;
  let Inst{13-8} = II{5-0};
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rd16;
  let Inst{19-16} = Rd16{3-0};
}
class Enc_16601956 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_15946706 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{6-5} = Ii{1-0};
  bits <3> Rdd8;
  let Inst{2-0} = Rdd8{2-0};
}
class Enc_6923828 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{13-13} = Ii{9-9};
  let Inst{10-8} = Ii{8-6};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
}
class Enc_1332717 : OpcodeHexagon {
  bits <2> Pu4;
  let Inst{6-5} = Pu4{1-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_1786883 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <6> Sdd64;
  let Inst{5-0} = Sdd64{5-0};
}
class Enc_14303394 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{8-5} = Ii{5-2};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_9282127 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-7} = Ii{7-2};
  bits <8> II;
  let Inst{13-13} = II{7-7};
  let Inst{6-0} = II{6-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_2813446 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{6-3} = Ii{3-0};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_364753 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
  bits <4> n1;
  let Inst{29-29} = n1{3-3};
  let Inst{26-25} = n1{2-1};
  let Inst{23-23} = n1{0-0};
}
class Enc_12477789 : OpcodeHexagon {
  bits <15> Ii;
  let Inst{21-21} = Ii{14-14};
  let Inst{13-13} = Ii{13-13};
  let Inst{11-1} = Ii{12-2};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
}
class Enc_44555 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_8497723 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{13-8} = Ii{5-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rxx32;
  let Inst{4-0} = Rxx32{4-0};
}
class Enc_4359901 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
  bits <4> n1;
  let Inst{29-29} = n1{3-3};
  let Inst{26-25} = n1{2-1};
  let Inst{22-22} = n1{0-0};
}
class Enc_11271630 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{6-3} = Ii{6-3};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_10501894 : OpcodeHexagon {
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <3> Rdd8;
  let Inst{2-0} = Rdd8{2-0};
}
class Enc_9768377 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{4-0} = Vd32{4-0};
}
class Enc_16268019 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{20-16} = Vuu32{4-0};
  bits <5> Vvv32;
  let Inst{12-8} = Vvv32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_8814718 : OpcodeHexagon {
  bits <18> Ii;
  let Inst{26-25} = Ii{17-16};
  let Inst{20-16} = Ii{15-11};
  let Inst{13-5} = Ii{10-2};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_6212930 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{8-5} = Ii{5-2};
  bits <2> Pt4;
  let Inst{10-9} = Pt4{1-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_5462762 : OpcodeHexagon {
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vv32;
  let Inst{12-8} = Vv32{4-0};
  bits <5> Vw32;
  let Inst{4-0} = Vw32{4-0};
}
class Enc_6154421 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{13-13} = Ii{6-6};
  let Inst{7-3} = Ii{5-1};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
}
class Enc_8940892 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_3531000 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{11-5} = Ii{6-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_14311138 : OpcodeHexagon {
  bits <5> Vuu32;
  let Inst{20-16} = Vuu32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
}
class Enc_2216485 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{22-21} = Ii{5-4};
  let Inst{13-13} = Ii{3-3};
  let Inst{7-5} = Ii{2-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_12395768 : OpcodeHexagon {
  bits <16> Ii;
  let Inst{26-25} = Ii{15-14};
  let Inst{20-16} = Ii{13-9};
  let Inst{13-13} = Ii{8-8};
  let Inst{7-0} = Ii{7-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
}
class Enc_11047413 : OpcodeHexagon {
  bits <6> II;
  let Inst{11-8} = II{5-2};
  let Inst{6-5} = II{1-0};
  bits <5> Ryy32;
  let Inst{4-0} = Ryy32{4-0};
  bits <5> Re32;
  let Inst{20-16} = Re32{4-0};
}
class Enc_1256611 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_7884306 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{8-4} = Ii{7-3};
}
class Enc_11244923 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{10-8} = Ii{9-7};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_8612939 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
  bits <5> n1;
  let Inst{29-29} = n1{4-4};
  let Inst{26-25} = n1{3-2};
  let Inst{22-22} = n1{1-1};
  let Inst{13-13} = n1{0-0};
}
class Enc_16355964 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{12-5} = Ii{7-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_12616482 : OpcodeHexagon {
  bits <6> II;
  let Inst{11-8} = II{5-2};
  let Inst{6-5} = II{1-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
  bits <5> Re32;
  let Inst{20-16} = Re32{4-0};
}
class Enc_5915771 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <5> n1;
  let Inst{28-28} = n1{4-4};
  let Inst{24-22} = n1{3-1};
  let Inst{8-8} = n1{0-0};
}
class Enc_14459927 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{10-8} = Ii{9-7};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_7504828 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{21-21} = Ii{9-9};
  let Inst{13-5} = Ii{8-0};
  bits <5> Ru32;
  let Inst{4-0} = Ru32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_14209223 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_3931661 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{8-5} = Ii{5-2};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_13606251 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{11-8} = Ii{5-2};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
}
class Enc_11475992 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vdd32;
  let Inst{7-3} = Vdd32{4-0};
}
class Enc_13133231 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_9959498 : OpcodeHexagon {
  bits <8> Ii;
  let Inst{22-21} = Ii{7-6};
  let Inst{13-13} = Ii{5-5};
  let Inst{7-5} = Ii{4-2};
  bits <5> Ru32;
  let Inst{4-0} = Ru32{4-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rd32;
  let Inst{12-8} = Rd32{4-0};
}
class Enc_8919369 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <5> n1;
  let Inst{28-28} = n1{4-4};
  let Inst{24-23} = n1{3-2};
  let Inst{13-13} = n1{1-1};
  let Inst{8-8} = n1{0-0};
}
class Enc_2968094 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{11-5} = Ii{6-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_4813442 : OpcodeHexagon {
  bits <6> Ii;
  let Inst{6-3} = Ii{5-2};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_4684887 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <4> Rs16;
  let Inst{19-16} = Rs16{3-0};
  bits <4> n1;
  let Inst{28-28} = n1{3-3};
  let Inst{25-23} = n1{2-0};
}
class Enc_15606259 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{11-8} = Ii{3-0};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
}
class Enc_2268028 : OpcodeHexagon {
  bits <3> Qtt8;
  let Inst{10-8} = Qtt8{2-0};
  bits <3> Qdd8;
  let Inst{5-3} = Qdd8{2-0};
}
class Enc_13430430 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Rt32;
  let Inst{20-16} = Rt32{4-0};
  bits <5> Vd32;
  let Inst{7-3} = Vd32{4-0};
  bits <3> Qxx8;
  let Inst{2-0} = Qxx8{2-0};
}
class Enc_13336212 : OpcodeHexagon {
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
  bits <1> n1;
  let Inst{9-9} = n1{0-0};
}
class Enc_15008287 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{20-16} = Vu32{4-0};
  bits <3> Rt8;
  let Inst{2-0} = Rt8{2-0};
  bits <5> Vx32;
  let Inst{7-3} = Vx32{4-0};
  bits <5> Vy32;
  let Inst{12-8} = Vy32{4-0};
}
class Enc_4897205 : OpcodeHexagon {
  bits <2> Qs4;
  let Inst{9-8} = Qs4{1-0};
  bits <2> Qd4;
  let Inst{1-0} = Qd4{1-0};
}
class Enc_8038806 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{11-8} = Ii{3-0};
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_12669374 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vxx32;
  let Inst{4-0} = Vxx32{4-0};
}
class Enc_971347 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{8-5} = Ii{3-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Ryy32;
  let Inst{4-0} = Ryy32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_1997594 : OpcodeHexagon {
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Rdd32;
  let Inst{4-0} = Rdd32{4-0};
}
class Enc_11940513 : OpcodeHexagon {
  bits <2> Ii;
  let Inst{13-13} = Ii{1-1};
  let Inst{7-7} = Ii{0-0};
  bits <2> Pv4;
  let Inst{6-5} = Pv4{1-0};
  bits <5> Rs32;
  let Inst{20-16} = Rs32{4-0};
  bits <5> Ru32;
  let Inst{12-8} = Ru32{4-0};
  bits <5> Rt32;
  let Inst{4-0} = Rt32{4-0};
}
class Enc_2735552 : OpcodeHexagon {
  bits <10> Ii;
  let Inst{10-8} = Ii{9-7};
  bits <2> Pv4;
  let Inst{12-11} = Pv4{1-0};
  bits <3> Os8;
  let Inst{2-0} = Os8{2-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_16410950 : OpcodeHexagon {
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <5> Vs32;
  let Inst{7-3} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_6226085 : OpcodeHexagon {
  bits <5> Ii;
  let Inst{12-8} = Ii{4-0};
  bits <5> II;
  let Inst{22-21} = II{4-3};
  let Inst{7-5} = II{2-0};
  bits <5> Rd32;
  let Inst{4-0} = Rd32{4-0};
}
class Enc_14193700 : OpcodeHexagon {
  bits <6> II;
  let Inst{5-0} = II{5-0};
  bits <3> Nt8;
  let Inst{10-8} = Nt8{2-0};
  bits <5> Re32;
  let Inst{20-16} = Re32{4-0};
}
class Enc_15763937 : OpcodeHexagon {
  bits <11> Ii;
  let Inst{21-20} = Ii{10-9};
  let Inst{7-1} = Ii{8-2};
  bits <3> Ns8;
  let Inst{18-16} = Ns8{2-0};
  bits <6> n1;
  let Inst{29-29} = n1{5-5};
  let Inst{26-25} = n1{4-3};
  let Inst{23-22} = n1{2-1};
  let Inst{13-13} = n1{0-0};
}
class Enc_2492727 : OpcodeHexagon {
  bits <5> Rss32;
  let Inst{20-16} = Rss32{4-0};
  bits <5> Rt32;
  let Inst{12-8} = Rt32{4-0};
  bits <2> Pd4;
  let Inst{1-0} = Pd4{1-0};
}
class Enc_13425035 : OpcodeHexagon {
  bits <2> Qv4;
  let Inst{12-11} = Qv4{1-0};
  bits <1> Mu2;
  let Inst{13-13} = Mu2{0-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_4135257 : OpcodeHexagon {
  bits <4> Ii;
  let Inst{10-8} = Ii{3-1};
  bits <4> Rs16;
  let Inst{7-4} = Rs16{3-0};
  bits <4> Rd16;
  let Inst{3-0} = Rd16{3-0};
}
class Enc_14631806 : OpcodeHexagon {
  bits <5> Vu32;
  let Inst{12-8} = Vu32{4-0};
  bits <5> Vdd32;
  let Inst{4-0} = Vdd32{4-0};
}
class Enc_12397062 : OpcodeHexagon {
  bits <9> Ii;
  let Inst{10-8} = Ii{8-6};
  bits <2> Qv4;
  let Inst{12-11} = Qv4{1-0};
  bits <5> Vs32;
  let Inst{4-0} = Vs32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
class Enc_11959851 : OpcodeHexagon {
  bits <7> Ii;
  let Inst{6-3} = Ii{6-3};
  bits <2> Pv4;
  let Inst{1-0} = Pv4{1-0};
  bits <5> Rtt32;
  let Inst{12-8} = Rtt32{4-0};
  bits <5> Rx32;
  let Inst{20-16} = Rx32{4-0};
}
