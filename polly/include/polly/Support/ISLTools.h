//===------ ISLTools.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tools, utilities, helpers and extensions useful in conjunction with the
// Integer Set Library (isl).
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_ISLTOOLS_H
#define POLLY_ISLTOOLS_H

#include "polly/Support/GICHelper.h"

namespace polly {

/// Return the range elements that are lexicographically smaller.
///
/// @param Map    { Space[] -> Scatter[] }
/// @param Strict True for strictly lexicographically smaller elements (exclude
///               same timepoints from the result).
///
/// @return { Space[] -> Scatter[] }
///         A map to all timepoints that happen before the timepoints the input
///         mapped to.
isl::map beforeScatter(isl::map Map, bool Strict);

/// Piecewise beforeScatter(isl::map,bool).
isl::union_map beforeScatter(isl::union_map UMap, bool Strict);

/// Return the range elements that are lexicographically larger.
///
/// @param Map    { Space[] -> Scatter[] }
/// @param Strict True for strictly lexicographically larger elements (exclude
///               same timepoints from the result).
///
/// @return { Space[] -> Scatter[] }
///         A map to all timepoints that happen after the timepoints the input
///         map originally mapped to.
isl::map afterScatter(isl::map Map, bool Strict);

/// Piecewise afterScatter(isl::map,bool).
isl::union_map afterScatter(const isl::union_map &UMap, bool Strict);

/// Construct a range of timepoints between two timepoints.
///
/// Example:
/// From := { A[] -> [0]; B[] -> [0] }
/// To   := {             B[] -> [10]; C[] -> [20] }
///
/// Result:
/// { B[] -> [i] : 0 < i < 10 }
///
/// Note that A[] and C[] are not in the result because they do not have a start
/// or end timepoint. If a start (or end) timepoint is not unique, the first
/// (respectively last) is chosen.
///
/// @param From     { Space[] -> Scatter[] }
///                 Map to start timepoints.
/// @param To       { Space[] -> Scatter[] }
///                 Map to end timepoints.
/// @param InclFrom Whether to include the start timepoints in the result. In
///                 the example, this would add { B[] -> [0] }
/// @param InclTo   Whether to include the end timepoints in the result. In this
///                 example, this would add { B[] -> [10] }
///
/// @return { Space[] -> Scatter[] }
///         A map for each domain element of timepoints between two extreme
///         points, or nullptr if @p From or @p To is nullptr, or the isl max
///         operations is exceeded.
isl::map betweenScatter(isl::map From, isl::map To, bool InclFrom, bool InclTo);

/// Piecewise betweenScatter(isl::map,isl::map,bool,bool).
isl::union_map betweenScatter(isl::union_map From, isl::union_map To,
                              bool InclFrom, bool InclTo);

/// If by construction a union map is known to contain only a single map, return
/// it.
///
/// This function combines isl_map_from_union_map() and
/// isl_union_map_extract_map(). isl_map_from_union_map() fails if the map is
/// empty because it does not know which space it would be in.
/// isl_union_map_extract_map() on the other hand does not check whether there
/// is (at most) one isl_map in the union, i.e. how it has been constructed is
/// probably wrong.
isl::map singleton(isl::union_map UMap, isl::space ExpectedSpace);

/// If by construction an isl_union_set is known to contain only a single
/// isl_set, return it.
///
/// This function combines isl_set_from_union_set() and
/// isl_union_set_extract_set(). isl_map_from_union_set() fails if the set is
/// empty because it does not know which space it would be in.
/// isl_union_set_extract_set() on the other hand does not check whether there
/// is (at most) one isl_set in the union, i.e. how it has been constructed is
/// probably wrong.
isl::set singleton(isl::union_set USet, isl::space ExpectedSpace);

/// Determine how many dimensions the scatter space of @p Schedule has.
///
/// The schedule must not be empty and have equal number of dimensions of any
/// subspace it contains.
///
/// The implementation currently returns the maximum number of dimensions it
/// encounters, if different, and 0 if none is encountered. However, most other
/// code will most likely fail if one of these happen.
unsigned getNumScatterDims(const isl::union_map &Schedule);

/// Return the scatter space of a @p Schedule.
///
/// This is basically the range space of the schedule map, but harder to
/// determine because it is an isl_union_map.
isl::space getScatterSpace(const isl::union_map &Schedule);

/// Construct an identity map for the given domain values.
///
/// There is no type resembling isl_union_space, hence we have to pass an
/// isl_union_set as the map's domain and range space.
///
/// @param USet           { Space[] }
///                       The returned map's domain and range.
/// @param RestrictDomain If true, the returned map only maps elements contained
///                       in @p USet and no other. If false, it returns an
///                       overapproximation with the identity maps of any space
///                       in @p USet, not just the elements in it.
///
/// @return { Space[] -> Space[] }
///         A map that maps each value of @p USet to itself.
isl::union_map makeIdentityMap(const isl::union_set &USet, bool RestrictDomain);

/// Reverse the nested map tuple in @p Map's domain.
///
/// @param Map { [Space1[] -> Space2[]] -> Space3[] }
///
/// @return { [Space2[] -> Space1[]] -> Space3[] }
isl::map reverseDomain(isl::map Map);

/// Piecewise reverseDomain(isl::map).
isl::union_map reverseDomain(const isl::union_map &UMap);

/// Add a constant to one dimension of a set.
///
/// @param Map    The set to shift a dimension in.
/// @param Pos    The dimension to shift. If negative, the dimensions are
///               counted from the end instead from the beginning. E.g. -1 is
///               the last dimension in the tuple.
/// @param Amount The offset to add to the specified dimension.
///
/// @return The modified set.
isl::set shiftDim(isl::set Set, int Pos, int Amount);

/// Piecewise shiftDim(isl::set,int,int).
isl::union_set shiftDim(isl::union_set USet, int Pos, int Amount);

/// Add a constant to one dimension of a map.
///
/// @param Map    The map to shift a dimension in.
/// @param Type   A tuple of @p Map which contains the dimension to shift.
/// @param Pos    The dimension to shift. If negative, the dimensions are
/// counted from the end instead from the beginning. Eg. -1 is the last
/// dimension in the tuple.
/// @param Amount The offset to add to the specified dimension.
///
/// @return The modified map.
isl::map shiftDim(isl::map Map, isl::dim Dim, int Pos, int Amount);

/// Add a constant to one dimension of a each map in a union map.
///
/// @param UMap   The maps to shift a dimension in.
/// @param Type   The tuple which contains the dimension to shift.
/// @param Pos    The dimension to shift. If negative, the dimensions are
///               counted from the ends of each map of union instead from their
///               beginning. E.g. -1 is the last dimension of any map.
/// @param Amount The offset to add to the specified dimension.
///
/// @return The union of all modified maps.
isl::union_map shiftDim(isl::union_map UMap, isl::dim Dim, int Pos, int Amount);

/// Simplify a set inplace.
void simplify(isl::set &Set);

/// Simplify a union set inplace.
void simplify(isl::union_set &USet);

/// Simplify a map inplace.
void simplify(isl::map &Map);

/// Simplify a union map inplace.
void simplify(isl::union_map &UMap);

/// Compute the reaching definition statement or the next overwrite for each
/// definition of an array element.
///
/// The reaching definition of an array element at a specific timepoint is the
/// statement instance that has written the current element's content.
/// Alternatively, this function determines for each timepoint and element which
/// write is going to overwrite an element at a future timepoint. This can be
/// seen as "reaching definition in reverse" where definitions are found in the
/// past.
///
/// For example:
///
/// Schedule := { Write[] -> [0]; Overwrite[] -> [10] }
/// Defs := { Write[] -> A[5]; Overwrite[] -> A[5] }
///
/// If index 5 of array A is written at timepoint 0 and 10, the resulting
/// reaching definitions are:
///
/// { [A[5] -> [i]] -> Write[] : 0 < i < 10;
///   [A[5] -> [i]] -> Overwrite[] : 10 < i }
///
/// Between timepoint 0 (Write[]) and timepoint 10 (Overwrite[]), the
/// content of A[5] is written by statement instance Write[] and after
/// timepoint 10 by Overwrite[]. Values not defined in the map have no known
/// definition. This includes the statement instance timepoints themselves,
/// because reads at those timepoints could either read the old or the new
/// value, defined only by the statement itself. But this can be changed by @p
/// InclPrevDef and @p InclNextDef. InclPrevDef=false and InclNextDef=true
/// returns a zone. Unless @p InclPrevDef and @p InclNextDef are both true,
/// there is only one unique definition per element and timepoint.
///
/// @param Schedule    { DomainWrite[] -> Scatter[] }
///                    Schedule of (at least) all array writes. Instances not in
///                    @p Writes are ignored.
/// @param Writes      { DomainWrite[] -> Element[] }
///                    Elements written to by the statement instances.
/// @param Reverse     If true, look for definitions in the future. That is,
///                    find the write that is overwrites the current value.
/// @param InclPrevDef Include the definition's timepoint to the set of
///                    well-defined elements (any load at that timepoint happen
///                    at the writes). In the example, enabling this option adds
///                    {[A[5] -> [0]] -> Write[]; [A[5] -> [10]] -> Overwrite[]}
///                    to the result.
/// @param InclNextDef Whether to assume that at the timepoint where an element
///                    is overwritten, it still contains the old value (any load
///                    at that timepoint would happen before the overwrite). In
///                    this example, enabling this adds
///                    { [A[] -> [10]] -> Write[] } to the result.
///
/// @return { [Element[] -> Scatter[]] -> DomainWrite[] }
///         The reaching definitions or future overwrite as described above, or
///         nullptr if either @p Schedule or @p Writes is nullptr, or the isl
///         max operations count has exceeded.
isl::union_map computeReachingWrite(isl::union_map Schedule,
                                    isl::union_map Writes, bool Reverse,
                                    bool InclPrevDef, bool InclNextDef);

/// Compute the timepoints where the contents of an array element are not used.
///
/// An element is unused at a timepoint when the element is overwritten in
/// the future, but it is not read in between. Another way to express this: the
/// time from when the element is written, to the most recent read before it, or
/// infinitely into the past if there is no read before. Such unused elements
/// can be overwritten by any value without changing the scop's semantics. An
/// example:
///
/// Schedule := { Read[] -> [0]; Write[] -> [10]; Def[] -> [20] }
/// Writes := { Write[] -> A[5]; Def[] -> A[6] }
/// Reads := { Read[] -> A[5] }
///
/// The result is:
///
/// { A[5] -> [i] : 0 < i < 10;
///   A[6] -> [i] : i < 20 }
///
/// That is, A[5] is unused between timepoint 0 (the read) and timepoint 10 (the
/// write). A[6] is unused before timepoint 20, but might be used after the
/// scop's execution (A[5] and any other A[i] as well). Use InclLastRead=false
/// and InclWrite=true to interpret the result as zone.
///
/// @param Schedule          { Domain[] -> Scatter[] }
///                          The schedule of (at least) all statement instances
///                          occurring in @p Writes or @p Reads. All other
///                          instances are ignored.
/// @param Writes            { DomainWrite[] -> Element[] }
///                          Elements written to by the statement instances.
/// @param Reads             { DomainRead[] -> Element[] }
///                          Elements read from by the statement instances.
/// @param ReadEltInSameInst Whether a load reads the value from a write
///                          that is scheduled at the same timepoint (Writes
///                          happen before reads). Otherwise, loads use the
///                          value of an element that it had before the
///                          timepoint (Reads before writes). For example:
///                          { Read[] -> [0]; Write[] -> [0] }
///                          With ReadEltInSameInst=false it is assumed that the
///                          read happens before the write, such that the
///                          element is never unused, or just at timepoint 0,
///                          depending on InclLastRead/InclWrite.
///                          With ReadEltInSameInst=false it assumes that the
///                          value just written is used. Anything before
///                          timepoint 0 is considered unused.
/// @param InclLastRead      Whether a timepoint where an element is last read
///                          counts as unused (the read happens at the beginning
///                          of its timepoint, and nothing (else) can use it
///                          during the timepoint). In the example, this option
///                          adds { A[5] -> [0] } to the result.
/// @param InclWrite         Whether the timepoint where an element is written
///                          itself counts as unused (the write happens at the
///                          end of its timepoint; no (other) operations uses
///                          the element during the timepoint). In this example,
///                          this adds
///                          { A[5] -> [10]; A[6] -> [20] } to the result.
///
/// @return { Element[] -> Scatter[] }
///         The unused timepoints as defined above, or nullptr if either @p
///         Schedule, @p Writes are @p Reads is nullptr, or the ISL max
///         operations count is exceeded.
isl::union_map computeArrayUnused(isl::union_map Schedule,
                                  isl::union_map Writes, isl::union_map Reads,
                                  bool ReadEltInSameInst, bool InclLastRead,
                                  bool InclWrite);

/// Convert a zone (range between timepoints) to timepoints.
///
/// A zone represents the time between (integer) timepoints, but not the
/// timepoints themselves. This function can be used to determine whether a
/// timepoint lies within a zone.
///
/// For instance, the range (1,3), representing the time between 1 and 3, is
/// represented by the zone
///
/// { [i] : 1 < i <= 3 }
///
/// The set of timepoints that lie completely within this range is
///
/// { [i] : 1 < i < 3 }
///
/// A typical use-case is the range in which a value written by a store is
/// available until it is overwritten by another value. If the write is at
/// timepoint 1 and its value is overwritten by another value at timepoint 3,
/// the value is available between those timepoints: timepoint 2 in this
/// example.
///
///
/// When InclStart is true, the range is interpreted left-inclusive, i.e. adds
/// the timepoint 1 to the result:
///
/// { [i] : 1 <= i < 3 }
///
/// In the use-case mentioned above that means that the value written at
/// timepoint 1 is already available in timepoint 1 (write takes place before
/// any read of it even if executed at the same timepoint)
///
/// When InclEnd is true, the range is interpreted right-inclusive, i.e. adds
/// the timepoint 3 to the result:
///
/// { [i] : 1 < i <= 3 }
///
/// In the use-case mentioned above that means that although the value is
/// overwritten in timepoint 3, the old value is still available at timepoint 3
/// (write takes place after any read even if executed at the same timepoint)
///
/// @param Zone      { Zone[] }
/// @param InclStart Include timepoints adjacent to the beginning of a zone.
/// @param InclEnd   Include timepoints adjacent to the ending of a zone.
///
/// @return { Scatter[] }
isl::union_set convertZoneToTimepoints(isl::union_set Zone, bool InclStart,
                                       bool InclEnd);

/// Like convertZoneToTimepoints(isl::union_set,InclStart,InclEnd), but convert
/// either the domain or the range of a map.
isl::union_map convertZoneToTimepoints(isl::union_map Zone, isl::dim Dim,
                                       bool InclStart, bool InclEnd);

/// Overload of convertZoneToTimepoints(isl::map,InclStart,InclEnd) to process
/// only a single map.
isl::map convertZoneToTimepoints(isl::map Zone, isl::dim Dim, bool InclStart,
                                 bool InclEnd);

/// Distribute the domain to the tuples of a wrapped range map.
///
/// @param Map { Domain[] -> [Range1[] -> Range2[]] }
///
/// @return { [Domain[] -> Range1[]] -> [Domain[] -> Range2[]] }
isl::map distributeDomain(isl::map Map);

/// Apply distributeDomain(isl::map) to each map in the union.
isl::union_map distributeDomain(isl::union_map UMap);

/// Prepend a space to the tuples of a map.
///
/// @param UMap   { Domain[] -> Range[] }
/// @param Factor { Factor[] }
///
/// @return { [Factor[] -> Domain[]] -> [Factor[] -> Range[]] }
isl::union_map liftDomains(isl::union_map UMap, isl::union_set Factor);

/// Apply a map to the 'middle' of another relation.
///
/// @param UMap { [DomainDomain[] -> DomainRange[]] -> Range[] }
/// @param Func { DomainRange[] -> NewDomainRange[] }
///
/// @return { [DomainDomain[] -> NewDomainRange[]] -> Range[] }
isl::union_map applyDomainRange(isl::union_map UMap, isl::union_map Func);

/// Intersect the range of @p Map with @p Range.
///
/// Since @p Map is an isl::map, the result will be a single space, even though
/// @p Range is an isl::union_set. This is the only difference to
/// isl::map::intersect_range and isl::union_map::interset_range.
///
/// @param Map   { Domain[] -> Range[] }
/// @param Range { Range[] }
///
/// @return { Domain[] -> Range[] }
isl::map intersectRange(isl::map Map, isl::union_set Range);

/// If @p PwAff maps to a constant, return said constant. If @p Max/@p Min, it
/// can also be a piecewise constant and it would return the minimum/maximum
/// value. Otherwise, return NaN.
isl::val getConstant(isl::pw_aff PwAff, bool Max, bool Min);

/// Dump a description of the argument to llvm::errs().
///
/// In contrast to isl's dump function, there are a few differences:
/// - Each polyhedron (pieces) is written on its own line.
/// - Spaces are sorted by structure. E.g. maps with same domain space are
///   grouped. Isl sorts them according to the space's hash function.
/// - Pieces of the same space are sorted using their lower bound.
/// - A more compact to_str representation is used instead of Isl's dump
///   functions that try to show the internal representation.
///
/// The goal is to get a better understandable representation that is also
/// useful to compare two sets. As all dump() functions, its intended use is to
/// be called in a debugger only.
///
/// isl_map_dump example:
/// [p_0, p_1, p_2] -> { Stmt0[i0] -> [o0, o1] : (o0 = i0 and o1 = 0 and i0 > 0
/// and i0 <= 5 - p_2) or (i0 = 0 and o0 = 0 and o1 = 0); Stmt3[i0] -> [o0, o1]
/// : (o0 = i0 and o1 = 3 and i0 > 0 and i0 <= 5 - p_2) or (i0 = 0 and o0 = 0
/// and o1 = 3); Stmt2[i0] -> [o0, o1] : (o0 = i0 and o1 = 1 and i0 >= 3 + p_0 -
/// p_1 and i0 > 0 and i0 <= 5 - p_2) or (o0 = i0 and o1 = 1 and i0 > 0 and i0
/// <= 5 - p_2 and i0 < p_0 - p_1) or (i0 = 0 and o0 = 0 and o1 = 1 and p_1 >= 3
/// + p_0) or (i0 = 0 and o0 = 0 and o1 = 1 and p_1 < p_0) or (p_0 = 0 and i0 =
/// 2 - p_1 and o0 = 2 - p_1 and o1 = 1 and p_2 <= 3 + p_1 and p_1 <= 1) or (p_1
/// = 1 + p_0 and i0 = 0 and o0 = 0 and o1 = 1) or (p_0 = 0 and p_1 = 2 and i0 =
/// 0 and o0 = 0 and o1 = 1) or (p_0 = -1 and p_1 = -1 and i0 = 0 and o0 = 0 and
/// o1 = 1); Stmt1[i0] -> [o0, o1] : (p_0 = -1 and i0 = 1 - p_1 and o0 = 1 - p_1
/// and o1 = 2 and p_2 <= 4 + p_1 and p_1 <= 0) or (p_0 = 0 and i0 = -p_1 and o0
/// = -p_1 and o1 = 2 and p_2 <= 5 + p_1 and p_1 < 0) or (p_0 = -1 and p_1 = 1
/// and i0 = 0 and o0 = 0 and o1 = 2) or (p_0 = 0 and p_1 = 0 and i0 = 0 and o0
/// = 0 and o1 = 2) }
///
/// dumpPw example (same set):
/// [p_0, p_1, p_2] -> {
///   Stmt0[0] -> [0, 0];
///   Stmt0[i0] -> [i0, 0] : 0 < i0 <= 5 - p_2;
///   Stmt1[0] -> [0, 2] : p_1 = 1 and p_0 = -1;
///   Stmt1[0] -> [0, 2] : p_1 = 0 and p_0 = 0;
///   Stmt1[1 - p_1] -> [1 - p_1, 2] : p_0 = -1 and p_1 <= 0 and p_2 <= 4 + p_1;
///   Stmt1[-p_1] -> [-p_1, 2] : p_0 = 0 and p_1 < 0 and p_2 <= 5 + p_1;
///   Stmt2[0] -> [0, 1] : p_1 >= 3 + p_0;
///   Stmt2[0] -> [0, 1] : p_1 < p_0;
///   Stmt2[0] -> [0, 1] : p_1 = 1 + p_0;
///   Stmt2[0] -> [0, 1] : p_1 = 2 and p_0 = 0;
///   Stmt2[0] -> [0, 1] : p_1 = -1 and p_0 = -1;
///   Stmt2[i0] -> [i0, 1] : i0 >= 3 + p_0 - p_1 and 0 < i0 <= 5 - p_2;
///   Stmt2[i0] -> [i0, 1] : 0 < i0 <= 5 - p_2 and i0 < p_0 - p_1;
///   Stmt2[2 - p_1] -> [2 - p_1, 1] : p_0 = 0 and p_1 <= 1 and p_2 <= 3 + p_1;
///   Stmt3[0] -> [0, 3];
///   Stmt3[i0] -> [i0, 3] : 0 < i0 <= 5 - p_2
/// }
/// @{
void dumpPw(const isl::set &Set);
void dumpPw(const isl::map &Map);
void dumpPw(const isl::union_set &USet);
void dumpPw(const isl::union_map &UMap);
void dumpPw(__isl_keep isl_set *Set);
void dumpPw(__isl_keep isl_map *Map);
void dumpPw(__isl_keep isl_union_set *USet);
void dumpPw(__isl_keep isl_union_map *UMap);
/// @}

/// Dump all points of the argument to llvm::errs().
///
/// Before being printed by dumpPw(), the argument's pieces are expanded to
/// contain only single points. If a dimension is unbounded, it keeps its
/// representation.
///
/// This is useful for debugging reduced cases where parameters are set to
/// constants to keep the example simple. Such sets can still contain
/// existential dimensions which makes the polyhedral hard to compare.
///
/// Example:
///   { [MemRef_A[i0] -> [i1]] : (exists (e0 = floor((1 + i1)/3): i0 = 1 and 3e0
///   <= i1 and 3e0 >= -1 + i1 and i1 >= 15 and i1 <= 25)) or (exists (e0 =
///   floor((i1)/3): i0 = 0 and 3e0 < i1 and 3e0 >= -2 + i1 and i1 > 0 and i1 <=
///   11)) }
///
/// dumpExpanded:
/// {
///   [MemRef_A[0] ->[1]];
///   [MemRef_A[0] ->[2]];
///   [MemRef_A[0] ->[4]];
///   [MemRef_A[0] ->[5]];
///   [MemRef_A[0] ->[7]];
///   [MemRef_A[0] ->[8]];
///   [MemRef_A[0] ->[10]];
///   [MemRef_A[0] ->[11]];
///   [MemRef_A[1] ->[15]];
///   [MemRef_A[1] ->[16]];
///   [MemRef_A[1] ->[18]];
///   [MemRef_A[1] ->[19]];
///   [MemRef_A[1] ->[21]];
///   [MemRef_A[1] ->[22]];
///   [MemRef_A[1] ->[24]];
///   [MemRef_A[1] ->[25]]
/// }
/// @{
void dumpExpanded(const isl::set &Set);
void dumpExpanded(const isl::map &Map);
void dumpExpanded(const isl::union_set &USet);
void dumpExpanded(const isl::union_map &UMap);
void dumpExpanded(__isl_keep isl_set *Set);
void dumpExpanded(__isl_keep isl_map *Map);
void dumpExpanded(__isl_keep isl_union_set *USet);
void dumpExpanded(__isl_keep isl_union_map *UMap);
/// @}

} // namespace polly

#endif /* POLLY_ISLTOOLS_H */
