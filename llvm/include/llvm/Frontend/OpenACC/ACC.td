//===-- ACC.td - OpenACC directive definition file ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for OpenACC 3.1 directives and clauses.
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
def ACCC_Async : Clause<"async"> {
  let flangClass = "ScalarIntExpr";
  let isValueOptional = true;
}

// 2.9.7
def ACCC_Auto : Clause<"auto"> {}

// 2.7.12
def ACCC_Attach : Clause<"attach"> {
  let flangClass = "AccObjectList";
}

// 2.15.1
def ACCC_Bind : Clause<"bind"> {
  let flangClass = "AccBindClause";
}

// 2.12
def ACCC_Capture : Clause<"capture"> {
}

// 2.9.1
def ACCC_Collapse : Clause<"collapse"> {
  let flangClass = "ScalarIntConstantExpr";
}

// 2.7.6
def ACCC_Copy : Clause<"copy"> {
  let flangClass = "AccObjectList";
}
// 2.7.7
def ACCC_Copyin : Clause<"copyin"> {
  let flangClass = "AccObjectListWithModifier";
}

// 2.7.8
def ACCC_Copyout : Clause<"copyout"> {
  let flangClass = "AccObjectListWithModifier";
}

// 2.7.9
def ACCC_Create : Clause<"create"> {
  let flangClass = "AccObjectListWithModifier";
}

// 2.5.15
def ACC_Default_none : ClauseVal<"none", 1, 1> { let isDefault = 1; }
def ACC_Default_present : ClauseVal<"present", 0, 1> {}

def ACCC_Default : Clause<"default"> {
  let flangClass = "AccDefaultClause";
  let enumClauseValue = "DefaultValue";
  let allowedClauseValues = [
    ACC_Default_present,
    ACC_Default_none
  ];
}

// 2.14.3
def ACCC_DefaultAsync : Clause<"default_async"> {
  let flangClass = "ScalarIntExpr";
}

// 2.7.11
def ACCC_Delete : Clause<"delete"> {
  let flangClass = "AccObjectList";
}

// 2.7.13
def ACCC_Detach : Clause<"detach"> {
  let flangClass = "AccObjectList";
}

// 2.14.4
def ACCC_Device : Clause<"device"> {
  let flangClass = "AccObjectList";
}

// 2.14.1 - 2.14.2
def ACCC_DeviceNum : Clause<"device_num">  {
  let flangClass = "ScalarIntExpr";
}

// 2.7.4
def ACCC_DevicePtr : Clause<"deviceptr"> {
  let flangClass = "AccObjectList";
}

// 2.13.1
def ACCC_DeviceResident : Clause<"device_resident"> {
  let flangClass = "AccObjectList";
}

// 2.4
def ACCC_DeviceType : Clause<"device_type"> {
  let flangClass = "ScalarIntExpr";
  let defaultValue = "*";
  let isValueOptional = true;
  let isValueList = true;
}

// 2.6.6
def ACCC_Finalize : Clause<"finalize"> {}

// 2.5.13
def ACCC_FirstPrivate : Clause<"firstprivate"> {
  let flangClass = "AccObjectList";
}

// 2.9.2
def ACCC_Gang : Clause<"gang"> {
  let flangClass = "AccGangArgument";
  let isValueOptional = true;
}

// 2.14.4
def ACCC_Host : Clause<"host"> {
  let flangClass = "AccObjectList";
}

// 2.5.5
def ACCC_If : Clause <"if"> {
  let flangClass = "ScalarLogicalExpr";
}

// 2.14.4
def ACCC_IfPresent : Clause<"if_present"> {}

// 2.9.6
def ACCC_Independent : Clause<"independent"> {}

// 2.13.3
def ACCC_Link : Clause<"link"> {
  let flangClass = "AccObjectList";
}

// 2.7.10
def ACCC_NoCreate : Clause<"no_create"> {
  let flangClass = "AccObjectList";
}

// 2.15.1
def ACCC_NoHost : Clause<"nohost"> {}

// 2.5.9
def ACCC_NumGangs : Clause<"num_gangs"> {
  let flangClass = "ScalarIntExpr";
}

// 2.5.10
def ACCC_NumWorkers : Clause<"num_workers"> {
  let flangClass = "ScalarIntExpr";
}

// 2.7.5
def ACCC_Present : Clause<"present"> {
  let flangClass = "AccObjectList";
}

// 2.5.12
def ACCC_Private : Clause<"private"> {
  let flangClass = "AccObjectList";
}

// 2.9.8
def ACCC_Tile : Clause <"tile"> {
  let flangClass = "AccTileExprList";
}

// 2.8.1
def ACCC_UseDevice : Clause <"use_device"> {
  let flangClass = "AccObjectList";
}

// 2.12
def ACCC_Read : Clause<"read"> {}

// 2.5.14
def ACCC_Reduction : Clause<"reduction"> {
  let flangClass = "AccObjectListWithReduction";
}

// 2.5.6
def ACCC_Self : Clause<"self"> {
  let flangClass = "AccSelfClause";
  let isValueOptional = true;
}

// 2.9.5
def ACCC_Seq : Clause<"seq"> {}

// 2.9.4
def ACCC_Vector : Clause<"vector"> {
  let flangClass = "ScalarIntExpr";
  let isValueOptional = true;
}

// 2.5.11
def ACCC_VectorLength : Clause<"vector_length"> {
  let flangClass = "ScalarIntExpr";
}

// 2.16.2
def ACCC_Wait : Clause<"wait"> {
  let flangClass = "AccWaitArgument";
  let isValueOptional = true;
}

// 2.9.3
def ACCC_Worker: Clause<"worker"> {
  let flangClass = "ScalarIntExpr";
  let isValueOptional = true;
}

// 2.12
def ACCC_Write : Clause<"write"> {}

def ACCC_Unknown : Clause<"unknown"> {
  let isDefault = true;
}

//===----------------------------------------------------------------------===//
// Definition of OpenACC directives
//===----------------------------------------------------------------------===//

// 2.12
def ACC_Atomic : Directive<"atomic"> {}

// 2.6.5
def ACC_Data : Directive<"data"> {
  let allowedOnceClauses = [
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_Default>
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
}

// 2.13
def ACC_Declare : Directive<"declare"> {
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
}

// 2.5.3
def ACC_Kernels : Directive<"kernels"> {
  let allowedClauses = [
    VersionedClause<ACCC_Attach>,
    VersionedClause<ACCC_Copy>,
    VersionedClause<ACCC_Copyin>,
    VersionedClause<ACCC_Copyout>,
    VersionedClause<ACCC_Create>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_NoCreate>,
    VersionedClause<ACCC_Present>,
    VersionedClause<ACCC_DevicePtr>,
    VersionedClause<ACCC_Wait>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_Default>,
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_NumGangs>,
    VersionedClause<ACCC_NumWorkers>,
    VersionedClause<ACCC_Self>,
    VersionedClause<ACCC_VectorLength>
  ];
}

// 2.5.1
def ACC_Parallel : Directive<"parallel"> {
  let allowedClauses = [
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
    VersionedClause<ACCC_Wait>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_Default>,
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_NumGangs>,
    VersionedClause<ACCC_NumWorkers>,
    VersionedClause<ACCC_Reduction>,
    VersionedClause<ACCC_Self>,
    VersionedClause<ACCC_VectorLength>
  ];
}

// 2.5.2
def ACC_Serial : Directive<"serial"> {
  // Spec line 950-951: clause is as for the parallel construct except that the
  // num_gangs, num_workers, and vector_length clauses are not permitted.
  let allowedClauses = [
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
    VersionedClause<ACCC_Wait>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_Default>,
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_Reduction>,
    VersionedClause<ACCC_Self>
  ];
}

// 2.9
def ACC_Loop : Directive<"loop"> {
  let allowedClauses = [
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_Private>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Collapse>,
    VersionedClause<ACCC_Gang>,
    VersionedClause<ACCC_Reduction>,
    VersionedClause<ACCC_Tile>,
    VersionedClause<ACCC_Vector>,
    VersionedClause<ACCC_Worker>
  ];
  let allowedExclusiveClauses = [
    VersionedClause<ACCC_Auto>,
    VersionedClause<ACCC_Independent>,
    VersionedClause<ACCC_Seq>
  ];
}

// 2.10
def ACC_Cache : Directive<"cache"> {}

// 2.14.1
def ACC_Init : Directive<"init"> {
  let allowedOnceClauses = [
    VersionedClause<ACCC_DeviceNum>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_If>
  ];
}

// 2.15.1
def ACC_Routine : Directive<"routine"> {
  let allowedOnceClauses = [
    VersionedClause<ACCC_Bind>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_NoHost>
  ];
  let requiredClauses = [
    VersionedClause<ACCC_Gang>,
    VersionedClause<ACCC_Seq>,
    VersionedClause<ACCC_Vector>,
    VersionedClause<ACCC_Worker>
  ];
}

// 2.14.3
def ACC_Set : Directive<"set"> {
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
}

// 2.14.2
def ACC_Shutdown : Directive<"shutdown"> {
  let allowedOnceClauses = [
    VersionedClause<ACCC_DeviceNum>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_If>
  ];
}

// 2.14.4
def ACC_Update : Directive<"update"> {
  let allowedClauses = [
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_Wait>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_IfPresent>
  ];
  let requiredClauses = [
    VersionedClause<ACCC_Device>,
    VersionedClause<ACCC_Host>,
    VersionedClause<ACCC_Self>
  ];
}

// 2.16.3
def ACC_Wait : Directive<"wait"> {
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_If>
  ];
}

// 2.14.6
def ACC_EnterData : Directive<"enter data"> {
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
}

// 2.14.7
def ACC_ExitData : Directive<"exit data"> {
  let allowedClauses = [
    VersionedClause<ACCC_Wait>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_Finalize>
  ];
  let requiredClauses = [
    VersionedClause<ACCC_Copyout>,
    VersionedClause<ACCC_Delete>,
    VersionedClause<ACCC_Detach>
  ];
}

// 2.8
def ACC_HostData : Directive<"host_data"> {
  let allowedOnceClauses = [
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_IfPresent>
  ];
  let requiredClauses = [
    VersionedClause<ACCC_UseDevice>
  ];
}

// 2.11
def ACC_KernelsLoop : Directive<"kernels loop"> {
  let allowedClauses = [
    VersionedClause<ACCC_Copy>,
    VersionedClause<ACCC_Copyin>,
    VersionedClause<ACCC_Copyout>,
    VersionedClause<ACCC_Create>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_NoCreate>,
    VersionedClause<ACCC_Present>,
    VersionedClause<ACCC_Private>,
    VersionedClause<ACCC_DevicePtr>,
    VersionedClause<ACCC_Attach>,
    VersionedClause<ACCC_Wait>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_Collapse>,
    VersionedClause<ACCC_Default>,
    VersionedClause<ACCC_Gang>,
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_NumGangs>,
    VersionedClause<ACCC_NumWorkers>,
    VersionedClause<ACCC_Reduction>,
    VersionedClause<ACCC_Self>,
    VersionedClause<ACCC_Tile>,
    VersionedClause<ACCC_Vector>,
    VersionedClause<ACCC_VectorLength>,
    VersionedClause<ACCC_Worker>
  ];
  let allowedExclusiveClauses = [
    VersionedClause<ACCC_Auto>,
    VersionedClause<ACCC_Independent>,
    VersionedClause<ACCC_Seq>
  ];
}

// 2.11
def ACC_ParallelLoop : Directive<"parallel loop"> {
  let allowedClauses = [
    VersionedClause<ACCC_Attach>,
    VersionedClause<ACCC_Copy>,
    VersionedClause<ACCC_Copyin>,
    VersionedClause<ACCC_Copyout>,
    VersionedClause<ACCC_Create>,
    VersionedClause<ACCC_DevicePtr>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_FirstPrivate>,
    VersionedClause<ACCC_NoCreate>,
    VersionedClause<ACCC_Present>,
    VersionedClause<ACCC_Private>,
    VersionedClause<ACCC_Tile>,
    VersionedClause<ACCC_Wait>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_Collapse>,
    VersionedClause<ACCC_Default>,
    VersionedClause<ACCC_Gang>,
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_NumGangs>,
    VersionedClause<ACCC_NumWorkers>,
    VersionedClause<ACCC_Reduction>,
    VersionedClause<ACCC_Self>,
    VersionedClause<ACCC_Vector>,
    VersionedClause<ACCC_VectorLength>,
    VersionedClause<ACCC_Worker>
  ];
  let allowedExclusiveClauses = [
    VersionedClause<ACCC_Auto>,
    VersionedClause<ACCC_Independent>,
    VersionedClause<ACCC_Seq>
  ];
}

// 2.11
def ACC_SerialLoop : Directive<"serial loop"> {
  let allowedClauses = [
    VersionedClause<ACCC_Attach>,
    VersionedClause<ACCC_Copy>,
    VersionedClause<ACCC_Copyin>,
    VersionedClause<ACCC_Copyout>,
    VersionedClause<ACCC_Create>,
    VersionedClause<ACCC_DevicePtr>,
    VersionedClause<ACCC_DeviceType>,
    VersionedClause<ACCC_FirstPrivate>,
    VersionedClause<ACCC_NoCreate>,
    VersionedClause<ACCC_Present>,
    VersionedClause<ACCC_Private>,
    VersionedClause<ACCC_Wait>
  ];
  let allowedOnceClauses = [
    VersionedClause<ACCC_Async>,
    VersionedClause<ACCC_Collapse>,
    VersionedClause<ACCC_Default>,
    VersionedClause<ACCC_Gang>,
    VersionedClause<ACCC_If>,
    VersionedClause<ACCC_Reduction>,
    VersionedClause<ACCC_Self>,
    VersionedClause<ACCC_Tile>,
    VersionedClause<ACCC_Vector>,
    VersionedClause<ACCC_Worker>
  ];
  let allowedExclusiveClauses = [
    VersionedClause<ACCC_Auto>,
    VersionedClause<ACCC_Independent>,
    VersionedClause<ACCC_Seq>
  ];
}

def ACC_Unknown : Directive<"unknown"> {
  let isDefault = true;
}
