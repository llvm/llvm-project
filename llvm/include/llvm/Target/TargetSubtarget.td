//===- TargetSubtarget.td - Target Independent Subtarget ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class Register;
class RegisterClass;

// Determine whether inlining is allowed if caller and callee differ in whether
// a feature is enabled.
class InlineBehavior;

// Arbitrary differences are allowed.
def InlineIgnore : InlineBehavior;
// A callee with this feature enabled cannot be inlined into a caller without
// it enabled.
def InlineDefault : InlineBehavior;
// A callee without this feature enabled cannot be inlined into a caller with
// it enabled.
def InlineInverse : InlineBehavior;
// The feature must be either enabled in both caller and callee, or disabled in
// both.
def InlineMustMatch : InlineBehavior;

//===----------------------------------------------------------------------===//
// SubtargetFeature - A characteristic of the chip set.
//
class SubtargetFeature<string n, string f, string v, string d,
                       list<SubtargetFeature> i = [],
                       InlineBehavior ib = InlineDefault> {
  // Name - Feature name. Used by command line (-mattr=) to determine the
  // appropriate target chip.
  //
  string Name = n;

  // FieldName - Field in XXXSubtarget to be set by feature.
  //
  string FieldName = f;

  // Value - Value the XXXSubtarget field to be set to by feature.
  //
  // A value of "true" or "false" implies the field is a bool. Otherwise,
  // it is assumed to be an integer. the integer value may be the name of an
  // enum constant. If multiple features use the same integer field, the
  // field will be set to the maximum value of all enabled features that
  // share the field.
  //
  string Value = v;

  // Desc - Feature description. Used by command line (-mattr=) to display help
  // information.
  //
  string Desc = d;

  // Implies - Features that this feature implies are present. If one of those
  // features isn't set, then this one shouldn't be set either.
  //
  list<SubtargetFeature> Implies = !sort(f, i, f.Name);

  InlineBehavior InlineBehavior = ib;
}

//===----------------------------------------------------------------------===//
// Pull in the common support for MCPredicate (portable scheduling predicates).
//
include "llvm/Target/TargetInstrPredicate.td"

//===----------------------------------------------------------------------===//
// Pull in the common support for scheduling
//
include "llvm/Target/TargetSchedule.td"

/// Predicates - These are extra conditionals which are turned into instruction
/// selector matching code. Currently each predicate is just a string.
class Predicate<code cond> {

  /// This is a C++ expression which can be evaluated to return a bool,
  /// representing whether the predicate is satisfied.
  ///
  /// The following variables are in scope:
  /// - `const GET_DAGISEL_BODY *this` representing the target's DAGToDAGISel
  ///   instance.
  code CondString = cond;

  /// AssemblerMatcherPredicate - If this feature can be used by the assembler
  /// matcher, this is true. Targets should set this by inheriting their
  /// feature from the AssemblerPredicate class in addition to Predicate.
  bit AssemblerMatcherPredicate = false;

  /// AssemblerCondDag - Set of subtarget features being tested used
  /// as alternative condition string used for assembler matcher. Must be used
  /// with (all_of) to indicate that all features must be present, or (any_of)
  /// to indicate that at least one must be. The required lack of presence of
  /// a feature can be tested using a (not) node including the feature.
  /// e.g. "(all_of ModeThumb)" is translated to "(Bits & ModeThumb) != 0".
  ///      "(all_of (not ModeThumb))" is translated to
  ///      "(Bits & ModeThumb) == 0".
  ///      "(all_of ModeThumb, FeatureThumb2)" is translated to
  ///      "(Bits & ModeThumb) != 0 && (Bits & FeatureThumb2) != 0".
  ///      "(any_of ModeTumb, FeatureThumb2)" is translated to
  ///      "(Bits & ModeThumb) != 0 || (Bits & FeatureThumb2) != 0".
  /// all_of and any_of cannot be combined in a single dag, instead multiple
  /// predicates can be placed onto Instruction definitions.
  dag AssemblerCondDag;

  /// PredicateName - User-level name to use for the predicate. Mainly for use
  /// in diagnostics such as missing feature errors in the asm matcher.
  string PredicateName = "";

  /// Setting this to '1' indicates that the predicate must be recomputed on
  /// every function change. Most predicates can leave this at '0'.
  ///
  /// Ignored by SelectionDAG, it always recomputes the predicate on every use.
  bit RecomputePerFunction = false;
}

// Operators for combining SubtargetFeatures in AssemblerPredicates
def any_of;
def all_of;

/// AssemblerPredicate - This is a Predicate that can be used when the assembler
/// matches instructions and aliases.
class AssemblerPredicate<dag cond, string name = ""> {
  // See `Predicate.AssemblerMatcherPredicate`
  bit AssemblerMatcherPredicate = true;
  // See: Predicate.AssemblerCondDag`
  dag AssemblerCondDag = cond;
  // See: Predicate.PredicateName
  string PredicateName = name;
}

//===----------------------------------------------------------------------===//
// Processor chip sets - These values represent each of the chip sets supported
// by the scheduler. Each Processor definition requires corresponding
// instruction itineraries.
//
class Processor<string n, ProcessorItineraries pi, list<SubtargetFeature> f,
                list<SubtargetFeature> tunef = []> {
  // Name - Chip set name. Used by command line (-mcpu=) to determine the
  // appropriate target chip.
  //
  string Name = n;

  // SchedModel - The machine model for scheduling and instruction cost.
  //
  SchedMachineModel SchedModel = NoSchedModel;

  // ProcItin - The scheduling information for the target processor.
  //
  ProcessorItineraries ProcItin = pi;

  // Features - list of
  list<SubtargetFeature> Features = f;

  // TuneFeatures - list of features for tuning for this CPU. If the target
  // supports -mtune, this should contain the list of features used to make
  // microarchitectural optimization decisions for a given processor. While
  // Features should contain the architectural features for the processor.
  list<SubtargetFeature> TuneFeatures = tunef;
}

// ProcessorModel allows subtargets to specify the more general
// SchedMachineModel instead if a ProcessorItinerary. Subtargets will
// gradually move to this newer form.
//
// Although this class always passes NoItineraries to the Processor
// class, the SchedMachineModel may still define valid Itineraries.
class ProcessorModel<string n, SchedMachineModel m, list<SubtargetFeature> f,
                     list<SubtargetFeature> tunef = []>
  : Processor<n, NoItineraries, f, tunef> {
  let SchedModel = m;
}

// Defines an alternate name 'n' for the processor 'alias', resolving
// to the same subtarget features and scheduling model.
class ProcessorAlias<string n, string alias> {
  string Name = n;
  string Alias = alias;
}
