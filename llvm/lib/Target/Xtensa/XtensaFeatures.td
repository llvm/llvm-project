//===----------------------------------------------------------------------===//
// Xtensa subtarget features.
//===----------------------------------------------------------------------===//

// Xtensa ISA extensions (Xtensa Options).
def FeatureDensity : SubtargetFeature<"density", "HasDensity", "true",
                                      "Enable Density instructions">;
def HasDensity : Predicate<"Subtarget->hasDensity()">,
                 AssemblerPredicate<(all_of FeatureDensity)>;

def FeatureWindowed : SubtargetFeature<"windowed", "HasWindowed", "true",
                                       "Enable Xtensa Windowed Register option">;
def HasWindowed : Predicate<"Subtarget->hasWindowed()">,
                  AssemblerPredicate<(all_of FeatureWindowed)>;

def FeatureBoolean          : SubtargetFeature<"bool", "HasBoolean", "true",
                                               "Enable Xtensa Boolean extension">;
def HasBoolean              : Predicate<"Subtarget->hasBoolean()">,
                                         AssemblerPredicate<(all_of FeatureBoolean)>;
