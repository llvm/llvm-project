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

def FeatureMAC16 : SubtargetFeature<"mac16", "HasMAC16", "true",
                                    "Enable Xtensa MAC16 instructions">;
def HasMAC16 : Predicate<"Subtarget->hasMAC16()">,
               AssemblerPredicate<(all_of FeatureMAC16)>;

def FeatureBoolean          : SubtargetFeature<"bool", "HasBoolean", "true",
                                               "Enable Xtensa Boolean extension">;
def HasBoolean              : Predicate<"Subtarget->hasBoolean()">,
                                         AssemblerPredicate<(all_of FeatureBoolean)>;

def FeatureLoop : SubtargetFeature<"loop", "HasLoop", "true",
                                   "Enable Xtensa Loop extension">;
def HasLoop : Predicate<"Subtarget->hasLoop()">,
              AssemblerPredicate<(all_of FeatureLoop)>;

def FeatureSEXT : SubtargetFeature<"sext", "HasSEXT", "true",
                                   "Enable Xtensa Sign Extend option">;
def HasSEXT : Predicate<"Subtarget->hasSEXT()">,
              AssemblerPredicate<(all_of FeatureSEXT)>;

def FeatureCLAMPS : SubtargetFeature<"clamps", "HasCLAMPS", "true",
                                     "Enable Xtensa CLAMPS option">;
def HasCLAMPS : Predicate<"Subtarget->hasCLAMPS()">,
                AssemblerPredicate<(all_of FeatureCLAMPS)>;

def FeatureNSA : SubtargetFeature<"nsa", "HasNSA", "true",
                                  "Enable Xtensa NSA option">;
def HasNSA : Predicate<"Subtarget->hasNSA()">,
             AssemblerPredicate<(all_of FeatureNSA)>;

def FeatureMINMAX : SubtargetFeature<"minmax", "HasMINMAX", "true",
                                     "Enable Xtensa MINMAX option">;
def HasMINMAX : Predicate<"Subtarget->hasMINMAX()">,
                AssemblerPredicate<(all_of FeatureMINMAX)>;
