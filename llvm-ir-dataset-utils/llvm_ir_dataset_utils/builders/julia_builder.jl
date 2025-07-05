"""
Calling convention: julia julia_builder.jl <package name>
After which you receive the system image, and a bitcode
  archive which is to be unpacked with `ar -x`.
"""

using Pkg;

# Rope in the ARGS given to Julia
for x in ARGS
    
    # Adding the Julia package
    try
        Pkg.add(x);
    catch e
        # line is buggy rn and just prints whenever
        # Error given when triggered alone: 
        #   TypeError: in using, expected Symbol, got a value of type Core.SlotNumber
        println("Package not found.");
    end

    try
        Pkg.test(x);
    catch e
        println("Testing package failed.");
    end
end
