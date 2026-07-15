//===- XeGPULayoutImpl.h - Layout utility functions ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_UTILS_XeGPULayoutImpl_H_
#define MLIR_DIALECT_XEGPU_UTILS_XeGPULayoutImpl_H_

#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/XeGPU/uArch/uArchCommon.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir {

class VectorType;
class OpOperand;
class OpResult;
class OpBuilder;
class ValueRange;
class TypeConverter;
class OpFoldResult;

namespace xegpu {
class DistributeLayoutAttr;
class LayoutAttr;
class TensorDescType;
} // namespace xegpu

namespace xegpu {

LogicalResult propagateLayouts(OpBuilder &builder, Operation *target,
                               LayoutKind layoutKind, unsigned indexBitWidth,
                               bool printOnly = false);

LogicalResult resolveLayoutConflicts(Operation *target);

/// Callable returning the propagated layout for a given Value, used by the
/// layout-propagation helpers below.
using GetLayoutFnTy = llvm::function_ref<DistributeLayoutAttr(Value)>;

/// Propagate layouts from a region branch op's region entry block arguments
/// back to its init operands. The block argument's layout is obtained via
/// `getLayoutOfValue`; the matching layout is then recorded on each init
/// operand that flows into that block argument (e.g. scf.for's iter_args
/// inits), and on tensor descriptor block argument types.
LogicalResult propagateRegionArgsToInits(RegionBranchOpInterface regionOp,
                                         GetLayoutFnTy getLayoutOfValue);

/// Attach layout attributes to all vector-type operands of operations within
/// the given operation's nested region. Reports an error if any vector operand
/// lacks a layout attribute.
bool recoverTemporaryLayouts(Operation *rootOp);

/// Removes the LayoutAttr for a given OpOperand or OpResult if it exists.
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, OpOperand> ||
                                      std::is_same_v<T, OpResult>>>
void removeLayoutAttr(const T &operandOrResult);

/// Removes the DistributeLayoutAttr for each OpOperand and OpResult of the
/// given operation if they exist. If the operation contains regions, it is also
/// applied recursively to the contained operations
void removeLayoutAttrs(Operation *op);

/// Removes the temporary layout attributes for each OpOperand and OpResult of
/// the given operation. Recursive for contained operations if the given
/// operation contains regions.
void removeTemporaryLayoutAttrs(Operation *op);

/// Updates the NamedAttribute sequence by dropping sg-layout and
/// sg-data information from any DistributeLayoutAttr found.
SmallVector<NamedAttribute>
dropSgLayoutAndDataOnAttrs(ArrayRef<NamedAttribute> attrs);

/// Updates the NamedAttribute sequence by dropping inst-data information from
/// any DistributeLayoutAttr found.
SmallVector<NamedAttribute> dropInstDataOnAttrs(ArrayRef<NamedAttribute> attrs);

/// Infers the source layout attribute for a broadcast operation given the
/// result layout attribute, result shape, and source shape.
DistributeLayoutAttr inferBroadcastSourceLayout(DistributeLayoutAttr resLayout,
                                                ArrayRef<int64_t> resShape,
                                                ArrayRef<int64_t> srcShape);

/// Infers the source layout attribute for a reduction operation given the
/// result layout attribute and reduced dims.
DistributeLayoutAttr
inferMultiReductionSourceLayout(DistributeLayoutAttr resLayout,
                                SmallVector<int64_t> reduceDims);

/// Infers the source layout attribute for a reduction operation given the
/// result layout attribute and reduced dims.
DistributeLayoutAttr inferReductionSourceLayout(DistributeLayoutAttr resLayout);

/// Infers the source layout attribute for a transpose operation given the
/// result layout attribute and permutation.
DistributeLayoutAttr inferTransposeSourceLayout(DistributeLayoutAttr resLayout,
                                                ArrayRef<int64_t> permutation);

/// Infers the source layout attribute for a bitcast operation given the
/// result layout attribute, result element type bitwidth, and source element
/// type bitwidth.
DistributeLayoutAttr inferBitCastSourceLayout(DistributeLayoutAttr resLayout,
                                              int resElemTyBitWidth,
                                              int srcElemTyBitWidth);

/// Infers the source layout attribute for an interleave operation given the
/// result layout attribute. Interleave doubles the innermost dimension size.
DistributeLayoutAttr
inferInterleaveSourceLayout(DistributeLayoutAttr resLayout);

/// Infers the source layout attribute for a deinterleave operation given the
/// result layout attribute. Deinterleave halves the innermost dimension size.
DistributeLayoutAttr
inferDeinterleaveSourceLayout(DistributeLayoutAttr resLayout);

/// Infers the source layout attribute for a shape cast operation given the
/// result layout attribute, result shape, and source shape.
DistributeLayoutAttr inferShapeCastSourceLayout(DistributeLayoutAttr resLayout,
                                                ArrayRef<int64_t> resShape,
                                                ArrayRef<int64_t> srcShape);

/// Infers the source layout attribute for an insert strided slice operation
/// given the result layout attribute, result shape, and source shape. Removes
/// leading dimensions from the result layout to match the source shape size.
DistributeLayoutAttr
inferInsertStridedSliceSourceLayout(DistributeLayoutAttr resLayout,
                                    ArrayRef<int64_t> resShape,
                                    ArrayRef<int64_t> srcShape);

/// Infers the source layout attribute for an insert operation.
/// using same logic as inferInsertStridedSliceSourceLayout
DistributeLayoutAttr inferInsertSourceLayout(DistributeLayoutAttr resLayout,
                                             ArrayRef<int64_t> resShape,
                                             ArrayRef<int64_t> srcShape);

/// Infers the source layout attribute for an extract operation. Adds
/// leading dimensions to the source layout to match the source shape size.
DistributeLayoutAttr inferExtractSourceLayout(DistributeLayoutAttr resLayout,
                                              ArrayRef<int64_t> resShape,
                                              ArrayRef<int64_t> srcShape);

/// Infers the layout attribute for mask and offset operand for Chunked load
/// and store, given the anchor layout attribute for the value being load/store.
DistributeLayoutAttr
inferMaskOffsetLayoutForScatterIO(DistributeLayoutAttr payloadLayout,
                                  int chunkSize);

/// Infers the source layout attribute for an operand using result layout
/// attribute
DistributeLayoutAttr
inferSourceLayoutFromResultForNonAnchorOp(OpOperand &operand,
                                          DistributeLayoutAttr resLayout);

/// Note on the `consumerLayout` argument used by the consumer-driven setup* /
/// complete* helpers below:
///
/// Layout propagation is a backward dataflow analysis, so a producer learns its
/// consumers' demands one at a time. The `consumerLayout` passed to these
/// helpers is the *single* layout that the first consumer to reach the producer
/// has requested (see `getConsumerLayoutAt`); these helpers do not pick among,
/// or merge, multiple consumers, and they do not reason about cost (e.g. a
/// consumer inside a loop vs. one outside). If a producer has several consumers
/// with conflicting layout demands, only the first-arriving one shapes the
/// producer's anchor layout here; any later, inconsistent consumer is left
/// as-is and reconciled afterwards by the layout conflict resolution process
/// (`ResolveLayoutConflicts`), which inserts a `convert_layout` op on that
/// edge. So these helpers can always assume exactly one (possibly null)
/// consumer layout to honor.

/// Sets up layout for Multi-Reduction operations by creating a SliceAttr for
/// the result.
///
/// This function first attempts to construct a source layout that, when
/// sliced along reduction dimensions, produces a result layout compatible
/// with the consumer's preferred layout. This minimizes data redistribution
/// overhead. The SliceAttr for the result is then created based on the
/// derived source layout and the specified reduction dimensions.
SliceAttr setupMultiReductionResultLayout(LayoutKind layoutKind,
                                          VectorType srcVectorTy,
                                          DistributeLayoutAttr consumerLayout,
                                          SmallVector<int64_t> reductionDims,
                                          int numSg, const uArch::uArch *uArch);

/// Sets up layout for Reduction operations by creating a SliceAttr for the
/// result.
SliceAttr setupReductionResultLayout(LayoutKind layoutKind,
                                     VectorType srcVectorTy,
                                     const uArch::uArch *uArch);

/// Setup the result layout attribute for a bitcast operation based on element
/// type bitwidths. This ensures the source layout can always be derived from
/// the result layout.
///
/// When casting from a narrower to a wider element type (srcElemTyBitWidth <
/// resElemTyBitWidth), the result layout's innermost dimension data sizes
/// (inst_data, lane_data) are scaled up by the bitwidth ratio. This maintains
/// the invariant that the source layout can be recovered by adjusting the
/// result layout based on bitwidth ratio of input vs output.
DistributeLayoutAttr setupBitCastResultLayout(
    LayoutKind layoutKind, VectorType srcVectorTy, VectorType resVectorTy,
    DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch);

/// Sets up the result layout for an interleave operation to ensure the source
/// layout can be safely derived. Interleave doubles the innermost dimension,
/// so the result layout must ensure that laneData is at least 2 (or a multiple
/// of 2), and instData must be divisible by innermostDimLaneLayout * 2.
DistributeLayoutAttr setupInterleaveResultLayout(
    LayoutKind layoutKind, VectorType srcVectorTy, VectorType resVectorTy,
    DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch);

/// Sets up the result layout for an insert strided slice operation.
/// Creates a result layout based on the specified layout kind (InstData or
/// Lane).
DistributeLayoutAttr setupInsertStridedSliceResultLayout(
    LayoutKind layoutKind, VectorType srcVectorTy, VectorType resVectorTy,
    DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch);

/// Sets up the anchor layout for a load gather operation.
DistributeLayoutAttr setupLoadGatherAnchorLayout(
    LayoutKind layoutKind, VectorType vectorTy, int contigChunkSize,
    DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch);

/// Sets up the anchor layout for load matrix operation.
DistributeLayoutAttr setupLoadMatrixAnchorLayout(
    LayoutKind layoutKind, VectorType vectorTy, int contigChunkSize,
    DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch);

/// Sets up the anchor layout for a store scatter operation.
/// `numSg` is only used for Subgroup-kind layouts.
DistributeLayoutAttr setupStoreScatterAnchorLayout(LayoutKind layoutKind,
                                                   VectorType vectorTy,
                                                   int contigChunkSize,
                                                   int numSg,
                                                   const uArch::uArch *uArch);

/// Sets up the anchor layout for a store matrix operation.
/// `numSg` is only used for Subgroup-kind layouts.
DistributeLayoutAttr setupStoreMatrixAnchorLayout(LayoutKind layoutKind,
                                                  VectorType vectorTy,
                                                  int contigChunkSize,
                                                  int numSg,
                                                  const uArch::uArch *uArch);

/// If the consumer layout has only inst_data (no lane_layout/lane_data),
/// completes it by running the corresponding scatter-style Lane-kind setup
/// rule with inst_data as the destination shape. The resulting lane info is
/// merged with the consumer's inst_data so downstream setup* paths see a
/// fully-populated layout.
/// Returns the layout unchanged when it is null, has no inst_data, or already
/// carries lane info; returns nullopt when the derived lane factorization does
/// not divide the user's inst_data (an invalid inst_data).
std::optional<DistributeLayoutAttr> completeScatterLoadLaneLayoutFromInstData(
    DistributeLayoutAttr userSpecifiedLayout,
    DistributeLayoutAttr consumerLayout, Type elemTy,
    const xegpu::uArch::LoadGatherInstruction *uArchInstruction,
    const int subgroupSize);

/// Like completeScatterLoadLaneLayoutFromInstData, but for scatter stores
/// (store_scatter / store_matrix). A store is a data sink: lane info is derived
/// purely from inst_data using the uArch's StoreScatter per-lane store width,
/// with no consumer layout to reuse.
std::optional<DistributeLayoutAttr> completeScatterStoreLaneLayoutFromInstData(
    DistributeLayoutAttr specifiedLayout, Type elemTy,
    const xegpu::uArch::StoreScatterInstruction *uArchInstruction,
    const int subgroupSize);

/// Completes a user-provided 2D-block store_nd / prefetch_nd anchor that has
/// only inst_data. These ops are data sinks, so lane info is derived purely
/// from inst_data using the shared BlockIOInstructionInterface; one helper
/// serves both store_nd and prefetch_nd.
std::optional<DistributeLayoutAttr> completeBlockStoreLaneLayoutFromInstData(
    DistributeLayoutAttr specifiedLayout, Type elemTy,
    const xegpu::uArch::BlockIOInstructionInterface *uArchInstruction,
    const int subgroupSize);

/// Like completeBlockStoreLaneLayoutFromInstData, but for load_nd. The consumer
/// layout supplies the transform / transpose / packing properties; the lane
/// factorization is recomputed from inst_data (load-side lane counts differ
/// from the consumer's).
std::optional<DistributeLayoutAttr> completeBlockLoadLaneLayoutFromInstData(
    DistributeLayoutAttr specifiedLayout, DistributeLayoutAttr consumerLayout,
    Type elemTy,
    const xegpu::uArch::BlockIOInstructionInterface *uArchInstruction,
    const int subgroupSize);

/// Sets up the anchor layout for a store_nd operation. StoreNd does not
/// consider a consumer layout (it is a data sink), and picks its layout from
/// uArch block parameters. `numSg` is only used for Subgroup-kind layouts.
DistributeLayoutAttr setupStoreNdAnchorLayout(LayoutKind layoutKind,
                                              VectorType vectorTy, int numSg,
                                              const uArch::uArch *uArch);

/// Sets up the anchor layout for a prefetch_nd operation. PrefetchNd has no
/// value result and thus no consumer; it picks its layout from uArch block
/// parameters. `numSg` is only used for Subgroup-kind layouts.
DistributeLayoutAttr setupPrefetchNdAnchorLayout(LayoutKind layoutKind,
                                                 TensorDescType tdescTy,
                                                 int numSg,
                                                 const uArch::uArch *uArch);

/// Sets up the anchor layout for a load_nd operation. LoadNd takes a
/// (downstream) consumer layout and validates it against uArch constraints;
/// when valid, the consumer's `inst_data` / `sg_layout` are honored.
/// Otherwise defaults derived from uArch block parameters are used.
/// `consumerLayout` must be presented. `numSg` is only used for Subgroup-kind
/// layouts when the consumer does not already provide an sg_layout.
DistributeLayoutAttr
setupLoadNdAnchorLayout(LayoutKind layoutKind, VectorType vectorTy,
                        DistributeLayoutAttr consumerLayout, int numSg,
                        const uArch::uArch *uArch);

/// Sets up the anchor layouts for a dpas operands (A, B, and C/D).
/// The numSg and consumerLayout (optional) are only used by sg layout creation.
std::optional<std::tuple<DistributeLayoutAttr, DistributeLayoutAttr,
                         DistributeLayoutAttr>>
setupDpasLayout(LayoutKind layoutKind, VectorType aTy, VectorType bTy,
                VectorType cdTy, DistributeLayoutAttr consumerLayout, int numSg,
                const uArch::uArch *uArch);

/// Sets up the anchor layouts for dpas_mx operands (A, B, C/D, A_scale, and
/// B_scale). The numSg and consumerLayout (optional) are only used by sg layout
/// creation. A_scale and B_scale are optional.
std::optional<
    std::tuple<DistributeLayoutAttr, DistributeLayoutAttr, DistributeLayoutAttr,
               DistributeLayoutAttr, DistributeLayoutAttr>>
setupDpasMxLayout(LayoutKind layoutKind, VectorType aTy, VectorType bTy,
                  VectorType cdTy, VectorType aScaleTy, VectorType bScaleTy,
                  DistributeLayoutAttr consumerLayout, int numSg,
                  const uArch::uArch *uArch);

/// Completes user-provided DPAS A/B/C-D anchors that carry only inst_data by
/// filling in lane_layout / lane_data derived from the operand shapes (mirrors
/// the InstData branch of setupDpasLayout). Returns nullopt if the uArch lacks
/// the matmul instruction.
std::optional<std::tuple<DistributeLayoutAttr, DistributeLayoutAttr,
                         DistributeLayoutAttr>>
completeDpasLaneLayoutFromInstData(DistributeLayoutAttr aLayout,
                                   DistributeLayoutAttr bLayout,
                                   DistributeLayoutAttr cdLayout,
                                   VectorType aTy, VectorType bTy,
                                   VectorType cdTy, const uArch::uArch *uArch);

/// Like completeDpasLaneLayoutFromInstData, but for dpas_mx: additionally
/// re-derives the A_scale / B_scale layouts from the completed A / B layouts.
std::optional<
    std::tuple<DistributeLayoutAttr, DistributeLayoutAttr, DistributeLayoutAttr,
               DistributeLayoutAttr, DistributeLayoutAttr>>
completeDpasMxLaneLayoutFromInstData(DistributeLayoutAttr aLayout,
                                     DistributeLayoutAttr bLayout,
                                     DistributeLayoutAttr cdLayout,
                                     VectorType aTy, VectorType bTy,
                                     VectorType cdTy, VectorType aScaleTy,
                                     VectorType bScaleTy,
                                     const uArch::uArch *uArch);

/// Gets the expected layout for a given consumer operand. This will check if
/// the owning operation of the consumer operand is one of the special layout
/// users and determine the expected layout accordingly.
DistributeLayoutAttr getConsumerLayoutAt(OpOperand &operand);

/// Returns true if `op` is safe and cheap to clone: it has no side effects,
/// no regions, and all of its operands are themselves trivially
/// rematerializable (e.g. `vector.step`, splat `arith.constant`, or
/// `vector.create_mask` whose operands are constants).
bool isTriviallyRematerializable(Operation *op);

} // namespace xegpu

} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_UTILS_XEGPUUTILS_H_
