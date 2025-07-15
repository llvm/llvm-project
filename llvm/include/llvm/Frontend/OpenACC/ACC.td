//===-- ACC.td - OpenACC directive definition file ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for OpenACC 3.3 directives and clauses.
//
//===----------------------------------------------------------------------===//

include "llvm/Frontend/Directive/DirectiveBase.td"

//===----------------------------------------------------------------------===//
// Definition of general OpenACC information
//===----------------------------------------------------------------------===//

def OpenACC : DirectiveLanguage {
  let name = "OpenACC";
  let cppNamespace = "acc"; // final namespace will be llvm::acc
  let directivePrefix = "ACCD_";
  let clausePrefix = "ACCC_";
  let makeEnumAvailableInNamespace = true;
  let enableBitmaskEnumInNamespace = true;
  let clauseEnumSetClass = "AccClauseSet";
  let flangClauseBaseClass = "AccClause";
}

//===----------------------------------------------------------------------===//
// Definition of OpenACC clauses
//===----------------------------------------------------------------------===//

// 2.16.1
def ACCC_Async : Clause<[Spelling<"async">]> {
  let flangClass = "ScalarIntExpr";
  let isValueOptional = true;
}

// 2.9.7
def ACCC_Auto : Clause<[Spelling<"auto">]> {}

// 2.7.12
def ACCC_Attach : Clause<[Spelling<"attach">]> {
  let flangClass = "AccObjectList";
}

// 2.15.1
def ACCC_Bind : Clause<[Spelling<"bind">]> {
  let flangClass = "AccBindClause";
}

// 2.12
def ACCC_Capture : Clause<[Spelling<"capture">]> {
}

// 2.9.1
def ACCC_Collapse : Clause<[Spelling<"collapse">]> {
  let flangClass = "AccCollapseArg";
}

// 2.7.6
def ACCC_Copy
    : Clause<[Spelling<"copy">, Spelling<"present_or_copy">,
              Spelling<"pcopy">]> {
  let flangClass = "AccObjectList";
}

// 2.7.7
def ACCC_Copyin : Clause<[Spelling<"copyin">, Spelling<"present_or_copyin">,
                          Spelling<"pcopyin">]> {
  let flangClass = "AccObjectListWithModifier";
  let clangAccSpelling = "CopyIn";
}

// 2.7.8
def ACCC_Copyout : Clause<[Spelling<"copyout">, Spelling<"present_or_copyout">,
                           Spelling<"pcopyout">]> {
  let flangClass = "AccObjectListWithModifier";
  let clangAccSpelling = "CopyOut";
}

// 2.7.9
def ACCC_Create : Clause<[Spelling<"create">, Spelling<"present_or_create">,
                          Spelling<"pcreate">]> {
  let flangClass = "AccObjectListWithModifier";
}

// 2.5.16
def ACC_Default_none : EnumVal<"none", 1, 1> { let isDefault = 1; }
def ACC_Default_present : EnumVal<"present", 0, 1> {}

def ACCC_Default : Clause<[Spelling<"default">]> {
  let flangClass = "AccDefaultClause";
  let enumClauseValue = "DefaultValue";
  let allowedClauseValues = [
    ACC_Default_present,
    ACC_Default_none
  ];
}

// 2.14.3
def ACCC_DefaultAsync : Clause<[Spelling<"default_async">]> {
  let flangClass = "ScalarIntExpr";
}

// 2.7.11
def ACCC_Delete : Clause<[Spelling<"delete">]> {
  let flangClass = "AccObjectList";
}

// 2.7.13
def ACCC_Detach : Clause<[Spelling<"detach">]> {
  let flangClass = "AccObjectList";
}

// 2.14.4
def ACCC_Device : Clause<[Spelling<"device">]> {
  let flangClass = "AccObjectList";
}

// 2.14.1 - 2.14.2
def ACCC_DeviceNum : Clause<[Spelling<"device_num">]>  {
  let flangClass = "ScalarIntExpr";
}

// 2.7.4
def ACCC_DevicePtr : Clause<[Spelling<"deviceptr">]> {
  let flangClass = "AccObjectList";
  let clangAccSpelling = "DevicePtr";
}

// 2.13.1
def ACCC_DeviceResident : Clause<[Spelling<"device_resident">]> {
  let flangClass = "AccObjectList";
}

// 2.4
def ACCC_DeviceType : Clause<[Spelling<"device_type">, Spelling<"dtype">]> {
  let flangClass = "AccDeviceTypeExprList";
  let defaultValue = "*";
}

// 2.6.6
def ACCC_Finalize : Clause<[Spelling<"finalize">]> {}

// 2.5.14
def ACCC_FirstPrivate : Clause<[Spelling<"firstprivate">]> {
  let flangClass = "AccObjectList";
  let clangAccSpelling = "FirstPrivate";
}

// 2.9.2
def ACCC_Gang : Clause<[Spelling<"gang">]> {
  let flangClass = "AccGangArgList";
  let isValueOptional = true;
}

// 2.14.4
def ACCC_Host : Clause<[Spelling<"host">]> {
  let flangClass = "AccObjectList";
}

// 2.5.6
def ACCC_If : Clause<[Spelling<"if">]> {
  let flangClass = "ScalarExpr";
}

// 2.14.4
def ACCC_IfPresent : Clause<[Spelling<"if_present">]> {}

// 2.9.6
def ACCC_Independent : Clause<[Spelling<"independent">]> {}

// 2.13.3
def ACCC_Link : Clause<[Spelling<"link">]> {
  let flangClass = "AccObjectList";
}

// 2.7.10
def ACCC_NoCreate : Clause<[Spelling<"no_create">]> {
  let flangClass = "AccObjectList";
}

// 2.15.1
def ACCC_NoHost : Clause<[Spelling<"nohost">]> {
  let clangAccSpelling = "NoHost";
}

// 2.5.10
def ACCC_NumGangs : Clause<[Spelling<"num_gangs">]> {
  let flangClass = "ScalarIntExpr";
  let isValueList = 1;
}

// 2.5.11
def ACCC_NumWorkers : Clause<[Spelling<"num_workers">]> {
  let flangClass = "ScalarIntExpr";
}

// 2.7.5
def ACCC_Present : Clause<[Spelling<"present">]> {
  let flangClass = "AccObjectList";
}

// 2.5.13
def ACCC_Private : Clause<[Spelling<"private">]> {
  let flangClass = "AccObjectList";
}

// 2.9.8
def ACCC_Tile : Clause<[Spelling<"tile">]> {
  let flangClass = "AccTileExprList";
}

// 2.8.1
def ACCC_UseDevice : Clause<[Spelling<"use_device">]> {
  let flangClass = "AccObjectList";
}

// 2.12
def ACCC_Read : Clause<[Spelling<"read">]> {}

// 2.5.15
def ACCC_Reduction : Clause<[Spelling<"reduction">]> {
  let flangClass = "AccObjectListWithReduction";
}

// 2.5.7
def ACCC_Self : Clause<[Spelling<"self">]> {
  let flangClass = "AccSelfClause";
  let isValueOptional = true;
}

// 2.9.5
def ACCC_Seq : Clause<[Spelling<"seq">]> {}

// Non-standard extension
def ACCC_ShortLoop : Clause<[Spelling<"shortloop">]> {}

// 2.9.4
def ACCC_Vector : Clause<[Spelling<"vector">]> {
  let flangClass = "ScalarIntExpr";
  let isValueOptional = true;
  let prefix = "length";
}

// 2.5.12
def ACCC_VectorLength : Clause<[Spelling<"vector_length">]> {
  let flangClass = "ScalarIntExpr";
}

// 2.16.2
def ACCC_Wait : Clause<[Spelling<"wait">]> {
  let flangClass = "AccWaitArgument";
  let isValueOptional = true;
}

// 2.9.3
def ACCC_Worker: Clause<[Spelling<"worker">]> {
  let flangClass = "ScalarIntExpr";
  let isValueOptional = true;
  let prefix = "num";
}

// 2.12
def ACCC_Write : Clause<[Spelling<"write">]> {}

def ACCC_Unknown : Clause<[Spelling<"unknown">]> {
  let isDefault = true;
}

//===----------------------------------------------------------------------===//
// Definition of OpenACC directives
//===----------------------------------------------------------------------===//

// 2.12
def ACC_Atomic : Directive<[Spelling<"atomic">]> {
  let allowedOnceClauses = [VersionedClause<ACCC_If, 34>];
  let association = AS_Block;
  let category = CA_Executable;
}

// 2.6.5
def ACC_Data : Directive<[Spelling<"data">]> {
  let allowedOnceClauses = [
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_Default>
  ];
  let allowedClauses = [
    VersionedClause<ACCC_Async, 32>,
    VersionedClause<ACCC_DeviceType, 32>,
    VersionedClause<ACCC_Wait, 32>
  ];
  let requiredClauses = [
    VersionedClause<ACCC_Attach>,
    VersionedClause<ACCC_Copy>,
    VersionedClause<ACCC_Copyin>,
    VersionedClause<ACCC_Copyout>,
    VersionedClause<ACCC_Create>,
    VersionedClause<ACCC_Default>,
    VersionedClause<ACCC_DevicePtr>,
    VersionedClause<ACCC_NoCreate>,
    VersionedClause<ACCC_Present>
  ];
  let association = AS_Block;
  let category = CA_Executable;
}

// 2.13
def ACC_Declare : Directive<[Spelling<"declare">]> {
  let allowedClauses = [
    VersionedClause<ACCC_Copy>,
    VersionedClause<ACCC_Copyin>,
    VersionedClause<ACCC_Copyout>,
    VersionedClause<ACCC_Create>,
    VersionedClause<ACCC_Present>,
    VersionedClause<ACCC_DevicePtr>,
    VersionedClause<ACCC_DeviceResident>,
    VersionedClause<ACCC_Link>
  ];
  let association = AS_None;
  let category = CA_Declarative;
}

// 2.5.3
def ACC_Kernels : Directive<[Spelling<"kernels">]> {
  let allowedClauses = [VersionedClause<ACCC_Async>,
                        VersionedClause<ACCC_Attach>,
                        VersionedClause<ACCC_Copy>,
                        VersionedClause<ACCC_Copyin>,
                        VersionedClause<ACCC_Copyout>,
                        VersionedClause<ACCC_Create>,
                        VersionedClause<ACCC_DeviceType>,
                        VersionedClause<ACCC_NoCreate>,
                        VersionedClause<ACCC_NumGangs>,
                        VersionedClause<ACCC_NumWorkers>,
                        VersionedClause<ACCC_Present>,
                        VersionedClause<ACCC_DevicePtr>,
                        VersionedClause<ACCC_VectorLength>,
                        VersionedClause<ACCC_Wait>];
  let allowedOnceClauses = [VersionedClause<ACCC_Default>,
                            VersionedClause<ACCC_If>,
                            VersionedClause<ACCC_Self>];
  let association = AS_Block;
  let category = CA_Executable;
}

// 2.5.1
def ACC_Parallel : Directive<[Spelling<"parallel">]> {
  let allowedClauses = [
    VersionedClause<ACCC_Attach>,
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_Copy>,
    VersionedClause<ACCC_Copyin>,
    VersionedClause<ACCC_Copyout>,
    VersionedClause<ACCC_Create>,
    VersionedClause<ACCC_DevicePtr>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_NoCreate>,
    VersionedClause<ACCC_NumGangs>,
    VersionedClause<ACCC_NumWorkers>,
    VersionedClause<ACCC_Present>,
    VersionedClause<ACCC_Private>,
    VersionedClause<ACCC_FirstPrivate>,
    VersionedClause<ACCC_Reduction>,
    VersionedClause<ACCC_Wait>,
    VersionedClause<ACCC_VectorLength>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Default>,
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_Self>
  ];
  let association = AS_Block;
  let category = CA_Executable;
}

// 2.5.2
def ACC_Serial : Directive<[Spelling<"serial">]> {
  // Spec line 950-951: clause is as for the parallel construct except that the
  // num_gangs, num_workers, and vector_length clauses are not permitted.
  let allowedClauses = [VersionedClause<ACCC_Async>,
                        VersionedClause<ACCC_Attach>,
                        VersionedClause<ACCC_Copy>,
                        VersionedClause<ACCC_Copyin>,
                        VersionedClause<ACCC_Copyout>,
                        VersionedClause<ACCC_Create>,
                        VersionedClause<ACCC_DevicePtr>,
                        VersionedClause<ACCC_DeviceType>,
                        VersionedClause<ACCC_NoCreate>,
                        VersionedClause<ACCC_Present>,
                        VersionedClause<ACCC_Private>,
                        VersionedClause<ACCC_FirstPrivate>,
                        VersionedClause<ACCC_Reduction>,
                        VersionedClause<ACCC_Wait>];
  let allowedOnceClauses = [VersionedClause<ACCC_Default>,
                            VersionedClause<ACCC_If>,
                            VersionedClause<ACCC_Self>];
  let association = AS_Block;
  let category = CA_Executable;
}

// 2.9
def ACC_Loop : Directive<[Spelling<"loop">]> {
  let allowedClauses = [
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_Private>,
    VersionedClause<ACCC_Reduction>,
    VersionedClause<ACCC_Collapse>,
    VersionedClause<ACCC_Gang>,
    VersionedClause<ACCC_ShortLoop>,
    VersionedClause<ACCC_Tile>,
    VersionedClause<ACCC_Vector>,
    VersionedClause<ACCC_Worker>
  ];
  let allowedExclusiveClauses = [
    VersionedClause<ACCC_Auto>,
    VersionedClause<ACCC_Independent>,
    VersionedClause<ACCC_Seq>
  ];
  let association = AS_Loop;
  let category = CA_Executable;
}

// 2.10
def ACC_Cache : Directive<[Spelling<"cache">]> {
  let association = AS_None;
  let category = CA_Executable;
}

// 2.14.1
def ACC_Init : Directive<[Spelling<"init">]> {
  let allowedOnceClauses = [VersionedClause<ACCC_DeviceNum>,
                            VersionedClause<ACCC_If>];
  let allowedClauses = [VersionedClause<ACCC_DeviceType>];
  let association = AS_None;
  let category = CA_Executable;
}

// 2.15.1
def ACC_Routine : Directive<[Spelling<"routine">]> {
  let allowedClauses = [
    VersionedClause<ACCC_Bind>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_Gang>,
    VersionedClause<ACCC_Seq>,
    VersionedClause<ACCC_Vector>,
    VersionedClause<ACCC_Worker>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_NoHost>
  ];
  let association = AS_Declaration;
  let category = CA_Declarative;
}

// 2.14.3
def ACC_Set : Directive<[Spelling<"set">]> {
  let allowedOnceClauses = [
    VersionedClause<ACCC_DefaultAsync>,
    VersionedClause<ACCC_DeviceNum>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_If>
  ];
  let requiredClauses = [
    // The three following clauses are also in allowedOnceClauses list due to
    // restriction 2255 - Two instances of the same clause may not appear on the
    // same directive.
    VersionedClause<ACCC_DefaultAsync>,
    VersionedClause<ACCC_DeviceNum>,
    VersionedClause<ACCC_DeviceType>
  ];
  let association = AS_None;
  let category = CA_Executable;
}

// 2.14.2
def ACC_Shutdown : Directive<[Spelling<"shutdown">]> {
  let allowedOnceClauses = [VersionedClause<ACCC_DeviceNum>,
                            VersionedClause<ACCC_If>];
  let allowedClauses = [VersionedClause<ACCC_DeviceType>];
  let association = AS_None;
  let category = CA_Executable;
}

// 2.14.4
def ACC_Update : Directive<[Spelling<"update">]> {
  let allowedClauses = [VersionedClause<ACCC_DeviceType>,
                        VersionedClause<ACCC_IfPresent>,
                        VersionedClause<ACCC_Wait>];
  let allowedOnceClauses = [VersionedClause<ACCC_Async>,
                            VersionedClause<ACCC_If>];
  let requiredClauses = [
    VersionedClause<ACCC_Device>,
    VersionedClause<ACCC_Host>,
    VersionedClause<ACCC_Self>
  ];
  let association = AS_None;
  let category = CA_Executable;
}

// 2.16.3
def ACC_Wait : Directive<[Spelling<"wait">]> {
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_If>
  ];
  let association = AS_None;
  let category = CA_Executable;
}

// 2.14.6
def ACC_EnterData : Directive<[Spelling<"enter data">]> {
  let allowedClauses = [
    VersionedClause<ACCC_Wait>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_If>
  ];
  let requiredClauses = [
    VersionedClause<ACCC_Attach>,
    VersionedClause<ACCC_Create>,
    VersionedClause<ACCC_Copyin>
  ];
  let association = AS_None;
  let category = CA_Executable;
}

// 2.14.7
def ACC_ExitData : Directive<[Spelling<"exit data">]> {
  let allowedClauses = [VersionedClause<ACCC_Finalize>,
                        VersionedClause<ACCC_Wait>];
  let allowedOnceClauses = [VersionedClause<ACCC_Async>,
                            VersionedClause<ACCC_If>];
  let requiredClauses = [
    VersionedClause<ACCC_Copyout>,
    VersionedClause<ACCC_Delete>,
    VersionedClause<ACCC_Detach>
  ];
  let association = AS_None;
  let category = CA_Executable;
}

// 2.8
def ACC_HostData : Directive<[Spelling<"host_data">]> {
  let allowedClauses = [VersionedClause<ACCC_IfPresent>];
  let allowedOnceClauses = [VersionedClause<ACCC_If>];
  let requiredClauses = [
    VersionedClause<ACCC_UseDevice>
  ];
  let association = AS_Block;
  let category = CA_Executable;
}

// 2.11
def ACC_KernelsLoop : Directive<[Spelling<"kernels loop">]> {
  let allowedClauses = [VersionedClause<ACCC_Async>,
                        VersionedClause<ACCC_Attach>,
                        VersionedClause<ACCC_Collapse>,
                        VersionedClause<ACCC_Copy>,
                        VersionedClause<ACCC_Copyin>,
                        VersionedClause<ACCC_Copyout>,
                        VersionedClause<ACCC_Create>,
                        VersionedClause<ACCC_DevicePtr>,
                        VersionedClause<ACCC_DeviceType>,
                        VersionedClause<ACCC_Gang>,
                        VersionedClause<ACCC_NoCreate>,
                        VersionedClause<ACCC_NumGangs>,
                        VersionedClause<ACCC_NumWorkers>,
                        VersionedClause<ACCC_Present>,
                        VersionedClause<ACCC_Private>,
                        VersionedClause<ACCC_Reduction>,
                        VersionedClause<ACCC_ShortLoop>,
                        VersionedClause<ACCC_Tile>,
                        VersionedClause<ACCC_Vector>,
                        VersionedClause<ACCC_VectorLength>,
                        VersionedClause<ACCC_Wait>,
                        VersionedClause<ACCC_Worker>];
  let allowedOnceClauses = [VersionedClause<ACCC_Default>,
                            VersionedClause<ACCC_If>,
                            VersionedClause<ACCC_Self>];
  let allowedExclusiveClauses = [
    VersionedClause<ACCC_Auto>,
    VersionedClause<ACCC_Independent>,
    VersionedClause<ACCC_Seq>
  ];
  let leafConstructs = [ACC_Kernels, ACC_Loop];
  let category = CA_Executable;
}

// 2.11
def ACC_ParallelLoop : Directive<[Spelling<"parallel loop">]> {
  let allowedClauses = [VersionedClause<ACCC_Async>,
                        VersionedClause<ACCC_Attach>,
                        VersionedClause<ACCC_Collapse>,
                        VersionedClause<ACCC_Copy>,
                        VersionedClause<ACCC_Copyin>,
                        VersionedClause<ACCC_Copyout>,
                        VersionedClause<ACCC_Create>,
                        VersionedClause<ACCC_DevicePtr>,
                        VersionedClause<ACCC_DeviceType>,
                        VersionedClause<ACCC_FirstPrivate>,
                        VersionedClause<ACCC_Gang>,
                        VersionedClause<ACCC_NoCreate>,
                        VersionedClause<ACCC_NumGangs>,
                        VersionedClause<ACCC_NumWorkers>,
                        VersionedClause<ACCC_Present>,
                        VersionedClause<ACCC_Private>,
                        VersionedClause<ACCC_Reduction>,
                        VersionedClause<ACCC_ShortLoop>,
                        VersionedClause<ACCC_Tile>,
                        VersionedClause<ACCC_Vector>,
                        VersionedClause<ACCC_VectorLength>,
                        VersionedClause<ACCC_Wait>,
                        VersionedClause<ACCC_Worker>];
  let allowedOnceClauses = [VersionedClause<ACCC_Default>,
                            VersionedClause<ACCC_If>,
                            VersionedClause<ACCC_Self>];
  let allowedExclusiveClauses = [
    VersionedClause<ACCC_Auto>,
    VersionedClause<ACCC_Independent>,
    VersionedClause<ACCC_Seq>
  ];
  let leafConstructs = [ACC_Parallel, ACC_Loop];
  let category = CA_Executable;
}

// 2.11
def ACC_SerialLoop : Directive<[Spelling<"serial loop">]> {
  let allowedClauses = [VersionedClause<ACCC_Async>,
                        VersionedClause<ACCC_Attach>,
                        VersionedClause<ACCC_Collapse>,
                        VersionedClause<ACCC_Copy>,
                        VersionedClause<ACCC_Copyin>,
                        VersionedClause<ACCC_Copyout>,
                        VersionedClause<ACCC_Create>,
                        VersionedClause<ACCC_DevicePtr>,
                        VersionedClause<ACCC_DeviceType>,
                        VersionedClause<ACCC_FirstPrivate>,
                        VersionedClause<ACCC_Gang>,
                        VersionedClause<ACCC_NoCreate>,
                        VersionedClause<ACCC_Present>,
                        VersionedClause<ACCC_Private>,
                        VersionedClause<ACCC_Reduction>,
                        VersionedClause<ACCC_ShortLoop>,
                        VersionedClause<ACCC_Tile>,
                        VersionedClause<ACCC_Vector>,
                        VersionedClause<ACCC_Wait>,
                        VersionedClause<ACCC_Worker>];
  let allowedOnceClauses = [VersionedClause<ACCC_Default>,
                            VersionedClause<ACCC_If>,
                            VersionedClause<ACCC_Self>];
  let allowedExclusiveClauses = [
    VersionedClause<ACCC_Auto>,
    VersionedClause<ACCC_Independent>,
    VersionedClause<ACCC_Seq>
  ];
  let leafConstructs = [ACC_Serial, ACC_Loop];
  let category = CA_Executable;
}

def ACC_Unknown : Directive<[Spelling<"unknown">]> {
  let isDefault = true;
  let association = AS_None;
  let category = CA_Utility;
}
