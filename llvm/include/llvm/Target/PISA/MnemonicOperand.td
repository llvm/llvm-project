//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DAG operator used for dag representation of a condition flag: (cflag FlagName value)
// These are used in EnumOptionOpnd to represent certain
def cflag;

// Function to find index of a item from a list of string.
// Note that tablegen does not allow recursive definition.
class getIndexFromList<list<string> slist, string item> {
  int Idx = !cond(!and(!ge(!size(slist),1), !eq(slist[0],item))  : 0,
                  !and(!ge(!size(slist),2), !eq(slist[1],item))  : 1,
                  !and(!ge(!size(slist),3), !eq(slist[2],item))  : 2,
                  !and(!ge(!size(slist),4), !eq(slist[3],item))  : 3,
                  !and(!ge(!size(slist),5), !eq(slist[4],item))  : 4,
                  !and(!ge(!size(slist),6), !eq(slist[5],item))  : 5,
                  !and(!ge(!size(slist),7), !eq(slist[6],item))  : 6,
                  !and(!ge(!size(slist),8), !eq(slist[7],item))  : 7,
                  !and(!ge(!size(slist),9), !eq(slist[8],item))  : 8,
                  !and(!ge(!size(slist),10),!eq(slist[9],item))  : 9,
                  !and(!ge(!size(slist),11),!eq(slist[10],item)) : 10,
                  !and(!ge(!size(slist),12),!eq(slist[11],item)) : 11,
                  !and(!ge(!size(slist),13),!eq(slist[12],item)) : 12,
                  !and(!ge(!size(slist),14),!eq(slist[13],item)) : 13,
                  !and(!ge(!size(slist),15),!eq(slist[14],item)) : 14,
                  !and(!ge(!size(slist),16),!eq(slist[15],item)) : 15,
                  true                                           : -1);
}

// Flags are encoded from msb - 1 - numflags to lsb
class getShiftedFlag<list<string> cFlagNames, string flagName> {
  defvar LsbIdx = !sub(!sub(!size(cFlagNames), 1), getIndexFromList<cFlagNames, flagName>.Idx);
  bits<16> enc = !shl(1, LsbIdx);
}

// Get bits enabled for all flags in daglist regardless of the value.
// e.g. if flag f1 is positioned at 2^2 and f2 in 2^0,
// [(cflag "f1", true), (cflag "f2", false)] -> {... 0, 1, 0, 1}
class EncodeCFlagMask<list<string> cFlagNames, list<dag> cFlags> {
  bits<16> enc = !foldl(0, cFlags, acc, fdag,
                        !or(getShiftedFlag<cFlagNames, !getdagarg<string>(fdag, "name")>.enc, acc));
}

// Get bits enables for all flags in daglist if the value is true.
// e.g. if flag f1 is positioned at 2^2 and f2 in 2^0,
// [(cflag "f1", true), (cflag "f2", false)] -> {... 0, 1, 0, 0}
class EncodeCFlag<list<string> cFlagNames, list<dag> cFlags> {
  bits<16> enc = !foldl(0, cFlags, acc, fdag,
                        !or(!and(!getdagarg<bit>(fdag, "val"),
                                 getShiftedFlag<cFlagNames, !getdagarg<string>(fdag, "name")>.enc),
                            acc));
}


// ----- Definitions for EnumOptions.
// Base definitions for operands that have a value selected from a
// set of limited options.

// Convert list of flag bits into a single accumulated value in bits<16>
class EncodeCFlagFromList<list<bit> bitList> {
  bits<16> enc = !foldl(0, bitList, acc, flag, !or(!shl(acc, 1), flag));
}

// Enum to list all type of EnumOptions
def EnumOptionClass: GenericEnum {
  let FilterClass = "EnumOptionClassEntry";
  let NameField = "EnumName";
}

// Lookup name string for enum types
def EnumOptionStrTable : GenericTable {
  let FilterClass = "EnumOptionClassEntry";
  let CppTypeName = "EnumOptionStrEntry";
  let Fields = ["Type", "Str"];
  string TypeOf_Type = "EnumOptionClass";
  let PrimaryKey = ["Type"];
  let PrimaryKeyName = "lookupEnumOptionStr";
}

class EnumOptionClassEntry<string name> {
  EnumOptionClassEntry Type = !cast<EnumOptionClassEntry>(NAME);
  string EnumName = "EO_"#name;
  string Str = name;
}

// Base class used to define per option enum entries
class EnumOptionEntry<EnumOptionClassEntry ClassID, string name, int val, string enumPrefix="", string optStr=!tolower(name), string enumName="", list<bit> cFlags=[]> {
  EnumOptionClassEntry OptClass = ClassID;
  EnumOptionEntry EnumValue = !cast<EnumOptionEntry>(NAME);
  string OptStr = optStr;

  // Capitalize the name and add prefix for enum type
  string EnumName = !cond(!not(!empty(enumName))  : enumName,
                          !empty(enumPrefix)      : !toupper(name),
                          true                    : enumPrefix#"_"#!toupper(name));

  // Need "bits<bit_size> Value" in all inheriting classes
  bits<32> Value = val;

  // Flags to classify the enum values
  bits<16> CFlagEnc = EncodeCFlagFromList<cFlags>.enc;
}

// Store EnumOption related info
class EnumOptionInfo<EnumOptionClassEntry OptID, ValueType vt, string enumPrefix,
                     bit isMandatory, int defaultVal,
                     Operand opnd, list<string> cFlagNames> {
   EnumOptionClassEntry OptClassID = OptID;
   ValueType VT = vt;
   string EnumPrefix = enumPrefix;
   bit IsMandatory = isMandatory;
   int DefaultVal = defaultVal;
   Operand Opnd = opnd;
   list<string> CFlagNames = cFlagNames;
}

class EnumOptionEntryDef<EnumOptionInfo info, string name, int val, string optStr=!tolower(name), string enumName="">
    : EnumOptionEntry<info.OptClassID, name, val, info.EnumPrefix, optStr, enumName>;

def EnumOptionTable : GenericTable {
  let FilterClass = "EnumOptionEntry";
  let Fields = ["OptClass", "Value", "OptStr", "CFlagEnc"];
  string TypeOf_OptClass = "EnumOptionClass";
  // FIXME: Is there a way to get EnumName printed in the table
  // rather than int literals?
  let PrimaryKey = ["OptClass", "Value"];
  let PrimaryKeyName = "lookupEnumOptionByValue";
}

def lookupEnumOptionByOptStr : SearchIndex {
  let Table = EnumOptionTable;
  let Key = ["OptClass", "OptStr"];
}

def lookupEnumOptionsWithClass : SearchIndex {
  let Table = EnumOptionTable;
  let Key = ["OptClass"];
}

// Base for defining enum and lookup tables
multiclass EnumTypeDef<string EnumEntryClassName=NAME#"Entry", string enumPrefix=""> {
  // Define enum definition
  // User must define a entry class named as EnumEntryClassName by inheriting EnumOptionEntry
  def NAME : GenericEnum {
    let FilterClass = EnumEntryClassName;
    let NameField = "EnumName";
    let ValueField = "Value";
    string EnumPrefix = enumPrefix;
  }
}

class EnumOptionOpndPMC<string name,
                        string optClassStr,
                        bit isMandatory = false,
                        int defaultVal = 0,
                        bits<16> CFlagMask = 0,
                        bits<16> CFlagCond = 0> : AsmOperandClass {
    let Name = name;
    let ParserMethod = "parseEnumOptionOpnd("#optClassStr#","#isMandatory#","#defaultVal#")";
    let ParserMethod = "parseEnumOptionOpnd("#optClassStr#","#isMandatory#","#CFlagMask#","#CFlagCond#","#defaultVal#")";
    let PredicateMethod = "isEnumOptionOpnd("#optClassStr#")";
    let RenderMethod = "addEnumOptionOpnd("#optClassStr#")";
}

class EnumOptionOpnd<ValueType vt,
                     string opndName,
                     EnumOptionClassEntry optClassID,
                     bit isMandatory = false,
                     int defaultVal = 0> : Operand<vt> {
  // Mask that need to filter relevant flags
  bits<16> CFlagMask = 0;
  // Matched value after masking
  bits<16> CFlagCond = 0;
  let OperandType = "OPERAND_IMMEDIATE";
  let PrintMethod = "printEnumOption("#optClassID.EnumName#","#isMandatory#","#defaultVal#")";
  let ParserMatchClass = EnumOptionOpndPMC<opndName, optClassID.EnumName, isMandatory, defaultVal, CFlagMask, CFlagCond>;
}

multiclass EnumOptionOpndDef<
           ValueType vt,
           string enumPrefix,
           bit isMandatory = false,
           int defaultVal = 0,
           list<string> cFlagNames = [],
           string EnumEntryClassName = NAME#"Entry"> {
  // Define enum type representing the enum class
  defvar TypeName = NAME#"ClassID";
  def TypeName : EnumOptionClassEntry<NAME>;
  defvar optClassID = !cast<EnumOptionClassEntry>(TypeName);
  defm NAME : EnumTypeDef<EnumEntryClassName, enumPrefix>;

  defvar OpndName = NAME#"Opnd";
  def OpndName
      : EnumOptionOpnd<vt, EnumEntryClassName#"Opnd", optClassID, isMandatory, defaultVal> ;
  def NAME#"EnumOptionInfo"
      : EnumOptionInfo<optClassID, vt, enumPrefix,
                       isMandatory, defaultVal,
                       !cast<Operand>(OpndName), cFlagNames>;
}

class EnumOptionIDEntry;

def EnumOptionID : GenericEnum {
  let FilterClass = "EnumOptionIDEntry";
}

class EnumOptionTableEntry<string optstr>{
  EnumOptionIDEntry OptID = !cast<EnumOptionIDEntry>(NAME);
  string OptStr = optstr;
}

class EnumOptionOpndParserMatchClass<string OptID> : AsmOperandClass {
  let Name = NAME;
  let ParserMethod = "parseEnumOptionOpnd("#OptID#")";
  let PredicateMethod = "isEnumOptionOpnd("#OptID#")";
  let RenderMethod = "addEnumOptionOpnd("#OptID#")";
}

class EnumOptionOpndWithDefaultOps<
    string OptID, EnumOptionOpndParserMatchClass pmc, bit isMandatory = false,
    int defaultVal = 0> :
        OperandWithDefaultOps<i32, (ops(i32 defaultVal))> {
  let OperandType = "OPERAND_IMMEDIATE";
  let MIOperandInfo = (ops i32imm);
  let PrintMethod = "printEnumOption("#OptID#","#isMandatory#","#defaultVal#")";
  let ParserMatchClass = pmc;
}

multiclass EnumOptionOpndWithDefaultOpsDef<
    string enumPrefix, bit isMandatory = false, int defaultVal = 0,
    string EnumEntryClassName = NAME#"Entry"> {
  defvar TypeName = NAME#"ClassID";
  def TypeName : EnumOptionClassEntry<NAME>;
  defvar optClassID = !cast<EnumOptionClassEntry>(TypeName);
  defm NAME : EnumTypeDef<EnumEntryClassName, enumPrefix>;

  defvar OptID = "EO_"#NAME;
  defvar OpndName = NAME#"Opnd";

  def NAME#"OpndPMC" : EnumOptionOpndParserMatchClass<OptID> {
    let ParserMethod = "parseEnumOptionOpnd("#OptID#","#isMandatory#",0,0,"#defaultVal#")";
  }

  def OpndName       : EnumOptionOpndWithDefaultOps<
    OptID, !cast<EnumOptionOpndParserMatchClass>(NAME#"OpndPMC"), isMandatory,
    defaultVal>;
}

// Selects subset of given enum types to be used as an operand type.
// cFlags are condition flags used to confine enum values to ones
// that matches given flag name and values.
// e.g. [(cflag "MyFlag1", true), (cflag "MyFlag2", false)] can be given
// to only select enum values with MyFlag1 = true and MyFlag2 = false.
class EnumOptionSubsetOpnd<EnumOptionInfo EOInfo, list<dag> cFlags>
    : EnumOptionOpnd<EOInfo.VT, NAME#"Opnd", EOInfo.OptClassID, EOInfo.IsMandatory, EOInfo.DefaultVal> {
  // This is used to mask only relevant flag bits this operand needs to match
  let CFlagMask = EncodeCFlagMask<EOInfo.CFlagNames, cFlags>.enc;
  // Flags values to be matched after masking.
  let CFlagCond = EncodeCFlag<EOInfo.CFlagNames, cFlags>.enc;
}

// EnableFlag: Tells which bool value is valid for the option
// 1: Accept only disabled, 2: Accept only enabled, 3: Accept both
class BoolOptionOpndParserMatchClass<string OptID, int EnableFlag=3> : AsmOperandClass {
  let Name = NAME;
  let ParserMethod = "parseBoolOptionOpnd("#OptID#", "#EnableFlag#")";
  let PredicateMethod = "isBoolOptionOpnd("#OptID#","#EnableFlag#")";
  let RenderMethod = "addBoolOptionOpnd("#OptID#")";
}

class BoolOptionOpnd<string OptID, BoolOptionOpndParserMatchClass pmc> : Operand<i16> {
  let OperandType = "OPERAND_IMMEDIATE";

  // Pass optString as name
  let PrintMethod = "printBoolOptionOpnd("#OptID#")";
  let ParserMatchClass = pmc;
}

class BoolOptionOpndWithDefaultOps<string OptID, BoolOptionOpndParserMatchClass pmc, bit defaultVal = 0> : OperandWithDefaultOps<i16, (ops(i16 defaultVal))> {
  let OperandType = "OPERAND_IMMEDIATE";

  // Pass optString as name
  let PrintMethod = "printBoolOptionOpnd("#OptID#")";
  let ParserMatchClass = pmc;
}

// Assign IDs for each bool options
def BoolOptionID : GenericEnum {
  let FilterClass = "BoolOptionIDEntry";
}

// Table with option id to option string mapping
def BoolOptionTable : GenericTable {
  let FilterClass  = "BoolOptionTableEntry";
  let Fields = ["OptID", "OptStr"];
  string TypeOf_OptID = "BoolOptionID";
  let PrimaryKey = ["OptID"];
  let PrimaryKeyName = "lookupBoolOptionByID";
}

def lookupBoolOptionByOptStr : SearchIndex {
  let Table = BoolOptionTable;
  let Key = ["OptStr"];
}

class BoolOptionIDEntry;

class BoolOptionTableEntry<string optstr>{
  BoolOptionIDEntry OptID = !cast<BoolOptionIDEntry>(NAME);
  string OptStr = optstr;
}

// Dummy entry to ensure BoolOptionTable is never empty
def BO_DUMMY_ : BoolOptionIDEntry, BoolOptionTableEntry<"">;

multiclass DefBoolOptionOpndType<string OptID, int EnableFlag = 3> {
  defvar PMCName = NAME#"OpndPMC";
  def PMCName : BoolOptionOpndParserMatchClass<OptID, EnableFlag>;
  def NAME#"Opnd" : BoolOptionOpnd<OptID, !cast<BoolOptionOpndParserMatchClass>(PMCName)>;
}

// DisabledOpnd type does not accept option but creates MIR operand with false value
// EnabledOpnd mandates optional field to be present and create true operand
multiclass BoolOptionOpndDef<string OptStr=!tolower(NAME), bit DefineDerivedOpnds = false> {
  def "BO_"#NAME : BoolOptionIDEntry, BoolOptionTableEntry<OptStr>;
  defvar OptID = "BO_"#NAME;
  defm NAME : DefBoolOptionOpndType<OptID, 3>;
  if DefineDerivedOpnds then {
    defm NAME#"Disabled" : DefBoolOptionOpndType<OptID, 1>;
    defm NAME#"Enabled" : DefBoolOptionOpndType<OptID, 2>;
  }
}

multiclass BoolOptionOpndWithDefaultOpsDef<string OptStr=!tolower(NAME), bit defaultVal = 0> {
  def "BO_"#NAME : BoolOptionIDEntry, BoolOptionTableEntry<OptStr>;
  defvar OptID = "BO_"#NAME;
  def NAME#"OpndPMC" : BoolOptionOpndParserMatchClass<OptID>;
  def NAME#"Opnd" : BoolOptionOpndWithDefaultOps<OptID, !cast<BoolOptionOpndParserMatchClass>(NAME#"OpndPMC"), defaultVal>;
}
