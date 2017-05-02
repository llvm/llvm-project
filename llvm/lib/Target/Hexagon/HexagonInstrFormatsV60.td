//==- HexagonInstrFormatsV60.td - Hexagon Instruction Formats -*- tablegen -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V60 instruction classes in TableGen format.
//
//===----------------------------------------------------------------------===//

//----------------------------------------------------------------------------//
//                         Instruction Classes Definitions +
//----------------------------------------------------------------------------//

class CVI_VA_Resource<dag outs, dag ins, string asmstr,
                       list<dag> pattern = [], string cstr = "",
                       InstrItinClass itin = CVI_VA>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VA>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VA_DV_Resource<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VA_DV>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VA_DV>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VX_Resource_long<dag outs, dag ins, string asmstr,
                       list<dag> pattern = [], string cstr = "",
                       InstrItinClass itin = CVI_VX_LONG>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VX>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VX_Resource_late<dag outs, dag ins, string asmstr,
                       list<dag> pattern = [], string cstr = "",
                       InstrItinClass itin = CVI_VX_LATE>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VX>,
     Requires<[HasV60T, UseHVX]>;

class CVI_VX_Resource<dag outs, dag ins, string asmstr,
                       list<dag> pattern = [], string cstr = "",
                       InstrItinClass itin = CVI_VX>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VX>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VX_DV_Resource<dag outs, dag ins, string asmstr,
                       list<dag> pattern = [], string cstr = "",
                       InstrItinClass itin = CVI_VX_DV>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VX_DV>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VX_DV_Slot2_Resource<dag outs, dag ins, string asmstr,
                       list<dag> pattern = [], string cstr = "",
                       InstrItinClass itin = CVI_VX_DV_SLOT2>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VX_DV>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VX_DV_Resource_long<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VX_DV_LONG>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VX_DV>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VP_Resource_long<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VP_LONG>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VP>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VP_VS_Resource_early<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VP_VS_EARLY>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VP_VS>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VP_VS_Resource_long<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VP_VS_LONG>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VP_VS>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VP_VS_Resource_long_early<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VP_VS_LONG_EARLY>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VP_VS>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VS_Resource<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VS>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VS>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VINLANESAT_Resource<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VINLANESAT>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VINLANESAT>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VS_Resource_long<dag outs, dag ins, string asmstr,
                           list<dag> pattern = [], string cstr = "",
                           InstrItinClass itin = CVI_VS>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VS>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_LD_Resource<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VM_LD>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_LD>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_LD_Resource_long<dag outs, dag ins, string asmstr,
                              list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VM_LD>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_LD>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_TMP_LD_Resource<dag outs, dag ins, string asmstr,
                             list<dag> pattern = [], string cstr = "",
                             InstrItinClass itin = CVI_VM_TMP_LD>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_TMP_LD>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_TMP_LD_Resource_long<dag outs, dag ins, string asmstr,
                                  list<dag> pattern = [], string cstr = "",
                                  InstrItinClass itin = CVI_VM_TMP_LD>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_TMP_LD>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_VP_LDU_Resource<dag outs, dag ins, string asmstr,
                             list<dag> pattern = [], string cstr = "",
                             InstrItinClass itin = CVI_VM_VP_LDU>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_VP_LDU>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_VP_LDU_Resource_long<dag outs, dag ins, string asmstr,
                                  list<dag> pattern = [], string cstr = "",
                                  InstrItinClass itin = CVI_VM_VP_LDU>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_VP_LDU>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_ST_Resource<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VM_ST>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_ST>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_ST_Resource_long<dag outs, dag ins, string asmstr,
                              list<dag> pattern = [], string cstr = "",
                              InstrItinClass itin = CVI_VM_ST>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_ST>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_NEW_ST_Resource<dag outs, dag ins, string asmstr,
                             list<dag> pattern = [], string cstr = "",
                             InstrItinClass itin = CVI_VM_NEW_ST>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_NEW_ST>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_NEW_ST_Resource_long<dag outs, dag ins, string asmstr,
                                  list<dag> pattern = [], string cstr = "",
                                  InstrItinClass itin = CVI_VM_NEW_ST>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_NEW_ST>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_STU_Resource<dag outs, dag ins, string asmstr,
                          list<dag> pattern = [], string cstr = "",
                          InstrItinClass itin = CVI_VM_STU>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_STU>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VM_STU_Resource_long<dag outs, dag ins, string asmstr,
                               list<dag> pattern = [], string cstr = "",
                               InstrItinClass itin = CVI_VM_STU>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VM_STU>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_HIST_Resource<dag outs, dag ins, string asmstr,
                        list<dag> pattern = [], string cstr = "",
                        InstrItinClass itin = CVI_HIST>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_HIST>,
     OpcodeHexagon, Requires<[HasV60T, UseHVX]>;

class CVI_VA_Resource1<dag outs, dag ins, string asmstr,
                       list<dag> pattern = [], string cstr = "",
                       InstrItinClass itin = CVI_VA>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VA>,
     Requires<[HasV60T, UseHVX]>;

class CVI_VX_DV_Resource1<dag outs, dag ins, string asmstr,
                         list<dag> pattern = [], string cstr = "",
                         InstrItinClass itin = CVI_VX_DV>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_VX_DV>,
     Requires<[HasV60T, UseHVX]>;

class CVI_HIST_Resource1<dag outs, dag ins, string asmstr,
                        list<dag> pattern = [], string cstr = "",
                        InstrItinClass itin = CVI_HIST>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCVI_HIST>,
     Requires<[HasV60T, UseHVX]>;
