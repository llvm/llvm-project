	OpCapability Kernel
	OpCapability Addresses
	OpCapability Int8
	OpCapability Linkage
	%1 = OpExtInstImport "OpenCL.std"
	OpMemoryModel Physical32 OpenCL
	OpSource OpenCL_CPP 100000
	OpName %33 "load_pointer"
	OpName %34 "store_pointer"
	OpName %35 "default_value"
	OpName %36 "store_object"
	OpName %37 "predicate"
	OpName %38 "foo"
	OpName %15 "_Z27__spirv_PredicatedLoadINTELPU3AS1Kibi"
	OpName %19 "_Z27__spirv_PredicatedLoadINTELPU3AS1Kibii"
	OpName %24 "_Z28__spirv_PredicatedStoreINTELPU3AS1Kiib"
	OpName %28 "_Z28__spirv_PredicatedStoreINTELPU3AS1Kiibi"
	OpName %2 "entry"
	OpDecorate %37 FuncParamAttr Zext
	OpDecorate %38 LinkageAttributes "foo" Export
	OpDecorate %15 LinkageAttributes "_Z27__spirv_PredicatedLoadINTELPU3AS1Kibi" Import
	OpDecorate %19 LinkageAttributes "_Z27__spirv_PredicatedLoadINTELPU3AS1Kibii" Import
	OpDecorate %24 LinkageAttributes "_Z28__spirv_PredicatedStoreINTELPU3AS1Kiib" Import
	OpDecorate %28 LinkageAttributes "_Z28__spirv_PredicatedStoreINTELPU3AS1Kiibi" Import
	%3 = OpTypeInt 32 0
	%4 = OpTypePointer CrossWorkgroup %3
	%5 = OpTypeBool
	%6 = OpTypeVoid
	%7 = OpTypeFunction %6 %4 %4 %3 %3 %5
	%8 = OpTypeInt 8 0
	%9 = OpTypePointer CrossWorkgroup %8
	%10 = OpTypeFunction %3 %9 %5 %3
	%11 = OpTypeFunction %3 %9 %5 %3 %3
	%12 = OpTypeFunction %6 %9 %3 %5
	%13 = OpTypeFunction %6 %9 %3 %5 %3
	%14 = OpConstantNull %3
	%15 = OpFunction %3 None %10
	%16 = OpFunctionParameter %9
	%17 = OpFunctionParameter %5
	%18 = OpFunctionParameter %3
	OpFunctionEnd
	%19 = OpFunction %3 None %11
	%20 = OpFunctionParameter %9
	%21 = OpFunctionParameter %5
	%22 = OpFunctionParameter %3
	%23 = OpFunctionParameter %3
	OpFunctionEnd
	%24 = OpFunction %6 None %12
	%25 = OpFunctionParameter %9
	%26 = OpFunctionParameter %3
	%27 = OpFunctionParameter %5
	OpFunctionEnd
	%28 = OpFunction %6 None %13
	%29 = OpFunctionParameter %9
	%30 = OpFunctionParameter %3
	%31 = OpFunctionParameter %5
	%32 = OpFunctionParameter %3
	OpFunctionEnd
	%38 = OpFunction %6 None %7             ; -- Begin function foo
	%33 = OpFunctionParameter %4
	%34 = OpFunctionParameter %4
	%35 = OpFunctionParameter %3
	%36 = OpFunctionParameter %3
	%37 = OpFunctionParameter %5
	%2 = OpLabel
	%39 = OpBitcast %9 %33
	%40 = OpFunctionCall %3 %15 %39 %37 %35
	%41 = OpBitcast %9 %33
	%42 = OpFunctionCall %3 %19 %41 %37 %35 %14
	%43 = OpBitcast %9 %34
	%44 = OpFunctionCall %6 %24 %43 %36 %37
	%45 = OpBitcast %9 %34
	%46 = OpFunctionCall %6 %28 %45 %36 %37 %14
	OpReturn
	OpFunctionEnd
                                        ; -- End function
