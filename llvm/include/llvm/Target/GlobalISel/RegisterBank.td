//===- RegisterBank.td - Register bank definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

/// Same as RegisterBankInfo::PartialMapping
class PartialMapping<int Len, int startIdx = 0> {
  int Length = Len;
  int StartIdx = startIdx;
}

class RegisterBank<string name, list<RegisterClass> classes> {
  string Name = name;
  list<RegisterClass> RegisterClasses = classes;

  // If some partial mappings can't be inferred from register class
  // add them to this list.
  list<PartialMapping> ExtraPartialMappings = [];

  // Exclude register sizes that are not used by the mapping policy.
  list<PartialMapping> IgnoredPartialMappings = [];

  // The fixed PartialMapping used for scalable values; zero disables them.
  int ScalableMappingSize = 0;
}

class MappingID<int id> {
  int ID = id;
}
def DefaultMappingID : MappingID<-1{31...0}>;
def InvalidMappingID : MappingID<-2{31...0}>;

// Assign a bank to a TypedOperand.
class RegBankTypeMapping<RegisterBank bank, TypedOperand type> {
  RegisterBank Bank = bank;
  TypedOperand Type = type;
}

class RegBankMappingRule<list<RegBankTypeMapping> typeMappings,
  int cost = 1, int priority = 0, id = DefaultMappingID, code predicate = [{}]> {
  list<RegBankTypeMapping> TypeMappings = typeMappings;
  int Priority = priority;
  MappingID ID = id;
  int Cost = cost;
  code Predicate = predicate;
  // TODO: support alternative mapping
  // bit IsAlternative = false;
}

class RegBankInstrMapping<list<GenericInstruction> opcodes,
                          list<RegBankMappingRule> rules> {
  assert !not(!empty(opcodes)), "no opcodes";
  assert !not(!empty(rules)), "no rules";
  list<GenericInstruction> Opcodes = opcodes;
  list<RegBankMappingRule> Rules = rules;
}
