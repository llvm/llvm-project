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

class RegBankMappingID<string name, int id> {
  string Name = name;
  int ID = id;
}
def RegBankDefaultMappingID : RegBankMappingID<"DefaultMappingID", -1{31...0}>;
def RegBankInvalidMappingID : RegBankMappingID<"InvalidMappingID", -2{31...0}>;

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

class RegBankMapping<RegisterBank bank> {
  RegisterBank Bank = bank;
}
class RegBankTypeMapping<RegisterBank bank, TypedOperand type> : RegBankMapping<bank> {
  TypedOperand Type = type;
}

// Define mapping rule for given instruction shape, each mapping in `Mappings` must
// cover all operands or TypedOperand types.
class RegBankMappingRule<list<RegBankMapping> mappings,
  int cost = 1, int priority = 0, RegBankMappingID id = RegBankDefaultMappingID, code predicate = [{}]> {
  list<RegBankMapping> Mappings = mappings;
  assert !foldl(1, mappings, cond, m,
    !and(cond, !isa<RegBankTypeMapping>(m))),
    "Mappings must not be plan RegBankMapping";
  int Priority = priority;
  RegBankMappingID ID = id;
  int Cost = cost;
  code Predicate = predicate;
  bit IsAlternative = false;
}
class RegBankAltMappingRule<list<RegBankMapping> mappings,
  int cost = 1, int priority = 0, RegBankMappingID id = RegBankDefaultMappingID, code predicate = [{}]>
    : RegBankMappingRule<mappings, cost, priority, id, predicate> {
  let IsAlternative = true;
}

/// Assign register bank mapping to instructions in `Opcodes`.
class RegBankInstrMapping<list<GenericInstruction> opcodes,
                          list<RegBankMappingRule> rules> {
  assert !not(!empty(opcodes)), "no opcodes";
  assert !not(!empty(rules)), "no rules";
  list<GenericInstruction> Opcodes = opcodes;
  list<RegBankMappingRule> Rules = rules;
}
