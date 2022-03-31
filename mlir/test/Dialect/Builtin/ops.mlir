// RUN: mlir-opt %s -allow-unregistered-dialect | mlir-opt -allow-unregistered-dialect

//===----------------------------------------------------------------------===//
// UnrealizedConversionCastOp
//===----------------------------------------------------------------------===//

%operand = "foo.op"() : () -> !foo.type
%tuple_operand = "foo.op"() : () -> !foo.tuple_type<!foo.type, !foo.type>

// An unrealized 0-1 conversion.
%result = unrealized_conversion_cast to !bar.tuple_type<>

// An unrealized 1-1 conversion.
%result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type

// An unrealized 1-N conversion.
%results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type

// An unrealized N-1 conversion.
%result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

// A basic 1D scalable vector
%scalable_vector_1d = "foo.op"() : () -> vector<[4]xi32>

// A 2D scalable vector
%scalable_vector_2d = "foo.op"() : () -> vector<[2x2]xf64>

// A 2D scalable vector with fixed-length dimensions
%scalable_vector_2d_mixed = "foo.op"() : () -> vector<2x[4]xbf16>

// A multi-dimensional vector with mixed scalable and fixed-length dimensions
%scalable_vector_multi_mixed = "foo.op"() : () -> vector<2x2x[4x4]xi8>
