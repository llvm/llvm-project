/*
 * kmp_affinity.cpp -- affinity management
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "kmp_affinity.h"
#include "kmp_i18n.h"
#include "kmp_io.h"
#include "kmp_str.h"
#include "kmp_wrapper_getpid.h"
#if KMP_USE_HIER_SCHED
#include "kmp_dispatch_hier.h"
#endif
#if KMP_USE_HWLOC
// Copied from hwloc
#define HWLOC_GROUP_KIND_INTEL_MODULE 102
#define HWLOC_GROUP_KIND_INTEL_TILE 103
#define HWLOC_GROUP_KIND_INTEL_DIE 104
#define HWLOC_GROUP_KIND_WINDOWS_PROCESSOR_GROUP 220
#endif
#include <ctype.h>

// The machine topology
kmp_topology_t *__kmp_topology = nullptr;
// KMP_HW_SUBSET environment variable
kmp_hw_subset_t *__kmp_hw_subset = nullptr;

// Store the real or imagined machine hierarchy here
static hierarchy_info machine_hierarchy;

void __kmp_cleanup_hierarchy() { machine_hierarchy.fini(); }

#if KMP_AFFINITY_SUPPORTED
// Helper class to see if place lists further restrict the fullMask
class kmp_full_mask_modifier_t {
  kmp_affin_mask_t *mask;

public:
  kmp_full_mask_modifier_t() {
    KMP_CPU_ALLOC(mask);
    KMP_CPU_ZERO(mask);
  }
  ~kmp_full_mask_modifier_t() {
    KMP_CPU_FREE(mask);
    mask = nullptr;
  }
  void include(const kmp_affin_mask_t *other) { KMP_CPU_UNION(mask, other); }
  // If the new full mask is different from the current full mask,
  // then switch them. Returns true if full mask was affected, false otherwise.
  bool restrict_to_mask() {
    // See if the new mask further restricts or changes the full mask
    if (KMP_CPU_EQUAL(__kmp_affin_fullMask, mask) || KMP_CPU_ISEMPTY(mask))
      return false;
    return __kmp_topology->restrict_to_mask(mask);
  }
};

static inline const char *
__kmp_get_affinity_env_var(const kmp_affinity_t &affinity,
                           bool for_binding = false) {
  if (affinity.flags.omp_places) {
    if (for_binding)
      return "OMP_PROC_BIND";
    return "OMP_PLACES";
  }
  return affinity.env_var;
}
#endif // KMP_AFFINITY_SUPPORTED

void __kmp_get_hierarchy(kmp_uint32 nproc, kmp_bstate_t *thr_bar) {
  kmp_uint32 depth;
  // The test below is true if affinity is available, but set to "none". Need to
  // init on first use of hierarchical barrier.
  if (TCR_1(machine_hierarchy.uninitialized))
    machine_hierarchy.init(nproc);

  // Adjust the hierarchy in case num threads exceeds original
  if (nproc > machine_hierarchy.base_num_threads)
    machine_hierarchy.resize(nproc);

  depth = machine_hierarchy.depth;
  KMP_DEBUG_ASSERT(depth > 0);

  thr_bar->depth = depth;
  __kmp_type_convert(machine_hierarchy.numPerLevel[0] - 1,
                     &(thr_bar->base_leaf_kids));
  thr_bar->skip_per_level = machine_hierarchy.skipPerLevel;
}

static int nCoresPerPkg, nPackages;
static int __kmp_nThreadsPerCore;
#ifndef KMP_DFLT_NTH_CORES
static int __kmp_ncores;
#endif

const char *__kmp_hw_get_catalog_string(kmp_hw_t type, bool plural) {
  switch (type) {
  case KMP_HW_SOCKET:
    return ((plural) ? KMP_I18N_STR(Sockets) : KMP_I18N_STR(Socket));
  case KMP_HW_DIE:
    return ((plural) ? KMP_I18N_STR(Dice) : KMP_I18N_STR(Die));
  case KMP_HW_MODULE:
    return ((plural) ? KMP_I18N_STR(Modules) : KMP_I18N_STR(Module));
  case KMP_HW_TILE:
    return ((plural) ? KMP_I18N_STR(Tiles) : KMP_I18N_STR(Tile));
  case KMP_HW_NUMA:
    return ((plural) ? KMP_I18N_STR(NumaDomains) : KMP_I18N_STR(NumaDomain));
  case KMP_HW_L3:
    return ((plural) ? KMP_I18N_STR(L3Caches) : KMP_I18N_STR(L3Cache));
  case KMP_HW_L2:
    return ((plural) ? KMP_I18N_STR(L2Caches) : KMP_I18N_STR(L2Cache));
  case KMP_HW_L1:
    return ((plural) ? KMP_I18N_STR(L1Caches) : KMP_I18N_STR(L1Cache));
  case KMP_HW_LLC:
    return ((plural) ? KMP_I18N_STR(LLCaches) : KMP_I18N_STR(LLCache));
  case KMP_HW_CORE:
    return ((plural) ? KMP_I18N_STR(Cores) : KMP_I18N_STR(Core));
  case KMP_HW_THREAD:
    return ((plural) ? KMP_I18N_STR(Threads) : KMP_I18N_STR(Thread));
  case KMP_HW_PROC_GROUP:
    return ((plural) ? KMP_I18N_STR(ProcGroups) : KMP_I18N_STR(ProcGroup));
  case KMP_HW_UNKNOWN:
  case KMP_HW_LAST:
    return KMP_I18N_STR(Unknown);
  }
  KMP_ASSERT2(false, "Unhandled kmp_hw_t enumeration");
  KMP_BUILTIN_UNREACHABLE;
}

const char *__kmp_hw_get_keyword(kmp_hw_t type, bool plural) {
  switch (type) {
  case KMP_HW_SOCKET:
    return ((plural) ? "sockets" : "socket");
  case KMP_HW_DIE:
    return ((plural) ? "dice" : "die");
  case KMP_HW_MODULE:
    return ((plural) ? "modules" : "module");
  case KMP_HW_TILE:
    return ((plural) ? "tiles" : "tile");
  case KMP_HW_NUMA:
    return ((plural) ? "numa_domains" : "numa_domain");
  case KMP_HW_L3:
    return ((plural) ? "l3_caches" : "l3_cache");
  case KMP_HW_L2:
    return ((plural) ? "l2_caches" : "l2_cache");
  case KMP_HW_L1:
    return ((plural) ? "l1_caches" : "l1_cache");
  case KMP_HW_LLC:
    return ((plural) ? "ll_caches" : "ll_cache");
  case KMP_HW_CORE:
    return ((plural) ? "cores" : "core");
  case KMP_HW_THREAD:
    return ((plural) ? "threads" : "thread");
  case KMP_HW_PROC_GROUP:
    return ((plural) ? "proc_groups" : "proc_group");
  case KMP_HW_UNKNOWN:
  case KMP_HW_LAST:
    return ((plural) ? "unknowns" : "unknown");
  }
  KMP_ASSERT2(false, "Unhandled kmp_hw_t enumeration");
  KMP_BUILTIN_UNREACHABLE;
}

const char *__kmp_hw_get_core_type_string(kmp_hw_core_type_t type) {
  switch (type) {
  case KMP_HW_CORE_TYPE_UNKNOWN:
  case KMP_HW_MAX_NUM_CORE_TYPES:
    return "unknown";
#if KMP_ARCH_X86 || KMP_ARCH_X86_64
  case KMP_HW_CORE_TYPE_ATOM:
    return "Intel Atom(R) processor";
  case KMP_HW_CORE_TYPE_CORE:
    return "Intel(R) Core(TM) processor";
#endif
  }
  KMP_ASSERT2(false, "Unhandled kmp_hw_core_type_t enumeration");
  KMP_BUILTIN_UNREACHABLE;
}

#if KMP_AFFINITY_SUPPORTED
// If affinity is supported, check the affinity
// verbose and warning flags before printing warning
#define KMP_AFF_WARNING(s, ...)                                                \
  if (s.flags.verbose || (s.flags.warnings && (s.type != affinity_none))) {    \
    KMP_WARNING(__VA_ARGS__);                                                  \
  }
#else
#define KMP_AFF_WARNING(s, ...) KMP_WARNING(__VA_ARGS__)
#endif

////////////////////////////////////////////////////////////////////////////////
// kmp_hw_thread_t methods
int kmp_hw_thread_t::compare_ids(const void *a, const void *b) {
  const kmp_hw_thread_t *ahwthread = (const kmp_hw_thread_t *)a;
  const kmp_hw_thread_t *bhwthread = (const kmp_hw_thread_t *)b;
  int depth = __kmp_topology->get_depth();
  for (int level = 0; level < depth; ++level) {
    if (ahwthread->ids[level] < bhwthread->ids[level])
      return -1;
    else if (ahwthread->ids[level] > bhwthread->ids[level])
      return 1;
  }
  if (ahwthread->os_id < bhwthread->os_id)
    return -1;
  else if (ahwthread->os_id > bhwthread->os_id)
    return 1;
  return 0;
}

#if KMP_AFFINITY_SUPPORTED
int kmp_hw_thread_t::compare_compact(const void *a, const void *b) {
  int i;
  const kmp_hw_thread_t *aa = (const kmp_hw_thread_t *)a;
  const kmp_hw_thread_t *bb = (const kmp_hw_thread_t *)b;
  int depth = __kmp_topology->get_depth();
  int compact = __kmp_topology->compact;
  KMP_DEBUG_ASSERT(compact >= 0);
  KMP_DEBUG_ASSERT(compact <= depth);
  for (i = 0; i < compact; i++) {
    int j = depth - i - 1;
    if (aa->sub_ids[j] < bb->sub_ids[j])
      return -1;
    if (aa->sub_ids[j] > bb->sub_ids[j])
      return 1;
  }
  for (; i < depth; i++) {
    int j = i - compact;
    if (aa->sub_ids[j] < bb->sub_ids[j])
      return -1;
    if (aa->sub_ids[j] > bb->sub_ids[j])
      return 1;
  }
  return 0;
}
#endif

void kmp_hw_thread_t::print() const {
  int depth = __kmp_topology->get_depth();
  printf("%4d ", os_id);
  for (int i = 0; i < depth; ++i) {
    printf("%4d ", ids[i]);
  }
  if (attrs) {
    if (attrs.is_core_type_valid())
      printf(" (%s)", __kmp_hw_get_core_type_string(attrs.get_core_type()));
    if (attrs.is_core_eff_valid())
      printf(" (eff=%d)", attrs.get_core_eff());
  }
  if (leader)
    printf(" (leader)");
  printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
// kmp_topology_t methods

// Add a layer to the topology based on the ids. Assume the topology
// is perfectly nested (i.e., so no object has more than one parent)
void kmp_topology_t::_insert_layer(kmp_hw_t type, const int *ids) {
  // Figure out where the layer should go by comparing the ids of the current
  // layers with the new ids
  int target_layer;
  int previous_id = kmp_hw_thread_t::UNKNOWN_ID;
  int previous_new_id = kmp_hw_thread_t::UNKNOWN_ID;

  // Start from the highest layer and work down to find target layer
  // If new layer is equal to another layer then put the new layer above
  for (target_layer = 0; target_layer < depth; ++target_layer) {
    bool layers_equal = true;
    bool strictly_above_target_layer = false;
    for (int i = 0; i < num_hw_threads; ++i) {
      int id = hw_threads[i].ids[target_layer];
      int new_id = ids[i];
      if (id != previous_id && new_id == previous_new_id) {
        // Found the layer we are strictly above
        strictly_above_target_layer = true;
        layers_equal = false;
        break;
      } else if (id == previous_id && new_id != previous_new_id) {
        // Found a layer we are below. Move to next layer and check.
        layers_equal = false;
        break;
      }
      previous_id = id;
      previous_new_id = new_id;
    }
    if (strictly_above_target_layer || layers_equal)
      break;
  }

  // Found the layer we are above. Now move everything to accommodate the new
  // layer. And put the new ids and type into the topology.
  for (int i = depth - 1, j = depth; i >= target_layer; --i, --j)
    types[j] = types[i];
  types[target_layer] = type;
  for (int k = 0; k < num_hw_threads; ++k) {
    for (int i = depth - 1, j = depth; i >= target_layer; --i, --j)
      hw_threads[k].ids[j] = hw_threads[k].ids[i];
    hw_threads[k].ids[target_layer] = ids[k];
  }
  equivalent[type] = type;
  depth++;
}

#if KMP_GROUP_AFFINITY
// Insert the Windows Processor Group structure into the topology
void kmp_topology_t::_insert_windows_proc_groups() {
  // Do not insert the processor group structure for a single group
  if (__kmp_num_proc_groups == 1)
    return;
  kmp_affin_mask_t *mask;
  int *ids = (int *)__kmp_allocate(sizeof(int) * num_hw_threads);
  KMP_CPU_ALLOC(mask);
  for (int i = 0; i < num_hw_threads; ++i) {
    KMP_CPU_ZERO(mask);
    KMP_CPU_SET(hw_threads[i].os_id, mask);
    ids[i] = __kmp_get_proc_group(mask);
  }
  KMP_CPU_FREE(mask);
  _insert_layer(KMP_HW_PROC_GROUP, ids);
  __kmp_free(ids);
}
#endif

// Remove layers that don't add information to the topology.
// This is done by having the layer take on the id = UNKNOWN_ID (-1)
void kmp_topology_t::_remove_radix1_layers() {
  int preference[KMP_HW_LAST];
  int top_index1, top_index2;
  // Set up preference associative array
  preference[KMP_HW_SOCKET] = 110;
  preference[KMP_HW_PROC_GROUP] = 100;
  preference[KMP_HW_CORE] = 95;
  preference[KMP_HW_THREAD] = 90;
  preference[KMP_HW_NUMA] = 85;
  preference[KMP_HW_DIE] = 80;
  preference[KMP_HW_TILE] = 75;
  preference[KMP_HW_MODULE] = 73;
  preference[KMP_HW_L3] = 70;
  preference[KMP_HW_L2] = 65;
  preference[KMP_HW_L1] = 60;
  preference[KMP_HW_LLC] = 5;
  top_index1 = 0;
  top_index2 = 1;
  while (top_index1 < depth - 1 && top_index2 < depth) {
    kmp_hw_t type1 = types[top_index1];
    kmp_hw_t type2 = types[top_index2];
    KMP_ASSERT_VALID_HW_TYPE(type1);
    KMP_ASSERT_VALID_HW_TYPE(type2);
    // Do not allow the three main topology levels (sockets, cores, threads) to
    // be compacted down
    if ((type1 == KMP_HW_THREAD || type1 == KMP_HW_CORE ||
         type1 == KMP_HW_SOCKET) &&
        (type2 == KMP_HW_THREAD || type2 == KMP_HW_CORE ||
         type2 == KMP_HW_SOCKET)) {
      top_index1 = top_index2++;
      continue;
    }
    bool radix1 = true;
    bool all_same = true;
    int id1 = hw_threads[0].ids[top_index1];
    int id2 = hw_threads[0].ids[top_index2];
    int pref1 = preference[type1];
    int pref2 = preference[type2];
    for (int hwidx = 1; hwidx < num_hw_threads; ++hwidx) {
      if (hw_threads[hwidx].ids[top_index1] == id1 &&
          hw_threads[hwidx].ids[top_index2] != id2) {
        radix1 = false;
        break;
      }
      if (hw_threads[hwidx].ids[top_index2] != id2)
        all_same = false;
      id1 = hw_threads[hwidx].ids[top_index1];
      id2 = hw_threads[hwidx].ids[top_index2];
    }
    if (radix1) {
      // Select the layer to remove based on preference
      kmp_hw_t remove_type, keep_type;
      int remove_layer, remove_layer_ids;
      if (pref1 > pref2) {
        remove_type = type2;
        remove_layer = remove_layer_ids = top_index2;
        keep_type = type1;
      } else {
        remove_type = type1;
        remove_layer = remove_layer_ids = top_index1;
        keep_type = type2;
      }
      // If all the indexes for the second (deeper) layer are the same.
      // e.g., all are zero, then make sure to keep the first layer's ids
      if (all_same)
        remove_layer_ids = top_index2;
      // Remove radix one type by setting the equivalence, removing the id from
      // the hw threads and removing the layer from types and depth
      set_equivalent_type(remove_type, keep_type);
      for (int idx = 0; idx < num_hw_threads; ++idx) {
        kmp_hw_thread_t &hw_thread = hw_threads[idx];
        for (int d = remove_layer_ids; d < depth - 1; ++d)
          hw_thread.ids[d] = hw_thread.ids[d + 1];
      }
      for (int idx = remove_layer; idx < depth - 1; ++idx)
        types[idx] = types[idx + 1];
      depth--;
    } else {
      top_index1 = top_index2++;
    }
  }
  KMP_ASSERT(depth > 0);
}

void kmp_topology_t::_set_last_level_cache() {
  if (get_equivalent_type(KMP_HW_L3) != KMP_HW_UNKNOWN)
    set_equivalent_type(KMP_HW_LLC, KMP_HW_L3);
  else if (get_equivalent_type(KMP_HW_L2) != KMP_HW_UNKNOWN)
    set_equivalent_type(KMP_HW_LLC, KMP_HW_L2);
#if KMP_MIC_SUPPORTED
  else if (__kmp_mic_type == mic3) {
    if (get_equivalent_type(KMP_HW_L2) != KMP_HW_UNKNOWN)
      set_equivalent_type(KMP_HW_LLC, KMP_HW_L2);
    else if (get_equivalent_type(KMP_HW_TILE) != KMP_HW_UNKNOWN)
      set_equivalent_type(KMP_HW_LLC, KMP_HW_TILE);
    // L2/Tile wasn't detected so just say L1
    else
      set_equivalent_type(KMP_HW_LLC, KMP_HW_L1);
  }
#endif
  else if (get_equivalent_type(KMP_HW_L1) != KMP_HW_UNKNOWN)
    set_equivalent_type(KMP_HW_LLC, KMP_HW_L1);
  // Fallback is to set last level cache to socket or core
  if (get_equivalent_type(KMP_HW_LLC) == KMP_HW_UNKNOWN) {
    if (get_equivalent_type(KMP_HW_SOCKET) != KMP_HW_UNKNOWN)
      set_equivalent_type(KMP_HW_LLC, KMP_HW_SOCKET);
    else if (get_equivalent_type(KMP_HW_CORE) != KMP_HW_UNKNOWN)
      set_equivalent_type(KMP_HW_LLC, KMP_HW_CORE);
  }
  KMP_ASSERT(get_equivalent_type(KMP_HW_LLC) != KMP_HW_UNKNOWN);
}

// Gather the count of each topology layer and the ratio
void kmp_topology_t::_gather_enumeration_information() {
  int previous_id[KMP_HW_LAST];
  int max[KMP_HW_LAST];

  for (int i = 0; i < depth; ++i) {
    previous_id[i] = kmp_hw_thread_t::UNKNOWN_ID;
    max[i] = 0;
    count[i] = 0;
    ratio[i] = 0;
  }
  int core_level = get_level(KMP_HW_CORE);
  for (int i = 0; i < num_hw_threads; ++i) {
    kmp_hw_thread_t &hw_thread = hw_threads[i];
    for (int layer = 0; layer < depth; ++layer) {
      int id = hw_thread.ids[layer];
      if (id != previous_id[layer]) {
        // Add an additional increment to each count
        for (int l = layer; l < depth; ++l)
          count[l]++;
        // Keep track of topology layer ratio statistics
        max[layer]++;
        for (int l = layer + 1; l < depth; ++l) {
          if (max[l] > ratio[l])
            ratio[l] = max[l];
          max[l] = 1;
        }
        // Figure out the number of different core types
        // and efficiencies for hybrid CPUs
        if (__kmp_is_hybrid_cpu() && core_level >= 0 && layer <= core_level) {
          if (hw_thread.attrs.is_core_eff_valid() &&
              hw_thread.attrs.core_eff >= num_core_efficiencies) {
            // Because efficiencies can range from 0 to max efficiency - 1,
            // the number of efficiencies is max efficiency + 1
            num_core_efficiencies = hw_thread.attrs.core_eff + 1;
          }
          if (hw_thread.attrs.is_core_type_valid()) {
            bool found = false;
            for (int j = 0; j < num_core_types; ++j) {
              if (hw_thread.attrs.get_core_type() == core_types[j]) {
                found = true;
                break;
              }
            }
            if (!found) {
              KMP_ASSERT(num_core_types < KMP_HW_MAX_NUM_CORE_TYPES);
              core_types[num_core_types++] = hw_thread.attrs.get_core_type();
            }
          }
        }
        break;
      }
    }
    for (int layer = 0; layer < depth; ++layer) {
      previous_id[layer] = hw_thread.ids[layer];
    }
  }
  for (int layer = 0; layer < depth; ++layer) {
    if (max[layer] > ratio[layer])
      ratio[layer] = max[layer];
  }
}

int kmp_topology_t::_get_ncores_with_attr(const kmp_hw_attr_t &attr,
                                          int above_level,
                                          bool find_all) const {
  int current, current_max;
  int previous_id[KMP_HW_LAST];
  for (int i = 0; i < depth; ++i)
    previous_id[i] = kmp_hw_thread_t::UNKNOWN_ID;
  int core_level = get_level(KMP_HW_CORE);
  if (find_all)
    above_level = -1;
  KMP_ASSERT(above_level < core_level);
  current_max = 0;
  current = 0;
  for (int i = 0; i < num_hw_threads; ++i) {
    kmp_hw_thread_t &hw_thread = hw_threads[i];
    if (!find_all && hw_thread.ids[above_level] != previous_id[above_level]) {
      if (current > current_max)
        current_max = current;
      current = hw_thread.attrs.contains(attr);
    } else {
      for (int level = above_level + 1; level <= core_level; ++level) {
        if (hw_thread.ids[level] != previous_id[level]) {
          if (hw_thread.attrs.contains(attr))
            current++;
          break;
        }
      }
    }
    for (int level = 0; level < depth; ++level)
      previous_id[level] = hw_thread.ids[level];
  }
  if (current > current_max)
    current_max = current;
  return current_max;
}

// Find out if the topology is uniform
void kmp_topology_t::_discover_uniformity() {
  int num = 1;
  for (int level = 0; level < depth; ++level)
    num *= ratio[level];
  flags.uniform = (num == count[depth - 1]);
}

// Set all the sub_ids for each hardware thread
void kmp_topology_t::_set_sub_ids() {
  int previous_id[KMP_HW_LAST];
  int sub_id[KMP_HW_LAST];

  for (int i = 0; i < depth; ++i) {
    previous_id[i] = -1;
    sub_id[i] = -1;
  }
  for (int i = 0; i < num_hw_threads; ++i) {
    kmp_hw_thread_t &hw_thread = hw_threads[i];
    // Setup the sub_id
    for (int j = 0; j < depth; ++j) {
      if (hw_thread.ids[j] != previous_id[j]) {
        sub_id[j]++;
        for (int k = j + 1; k < depth; ++k) {
          sub_id[k] = 0;
        }
        break;
      }
    }
    // Set previous_id
    for (int j = 0; j < depth; ++j) {
      previous_id[j] = hw_thread.ids[j];
    }
    // Set the sub_ids field
    for (int j = 0; j < depth; ++j) {
      hw_thread.sub_ids[j] = sub_id[j];
    }
  }
}

void kmp_topology_t::_set_globals() {
  // Set nCoresPerPkg, nPackages, __kmp_nThreadsPerCore, __kmp_ncores
  int core_level, thread_level, package_level;
  package_level = get_level(KMP_HW_SOCKET);
#if KMP_GROUP_AFFINITY
  if (package_level == -1)
    package_level = get_level(KMP_HW_PROC_GROUP);
#endif
  core_level = get_level(KMP_HW_CORE);
  thread_level = get_level(KMP_HW_THREAD);

  KMP_ASSERT(core_level != -1);
  KMP_ASSERT(thread_level != -1);

  __kmp_nThreadsPerCore = calculate_ratio(thread_level, core_level);
  if (package_level != -1) {
    nCoresPerPkg = calculate_ratio(core_level, package_level);
    nPackages = get_count(package_level);
  } else {
    // assume one socket
    nCoresPerPkg = get_count(core_level);
    nPackages = 1;
  }
#ifndef KMP_DFLT_NTH_CORES
  __kmp_ncores = get_count(core_level);
#endif
}

kmp_topology_t *kmp_topology_t::allocate(int nproc, int ndepth,
                                         const kmp_hw_t *types) {
  kmp_topology_t *retval;
  // Allocate all data in one large allocation
  size_t size = sizeof(kmp_topology_t) + sizeof(kmp_hw_thread_t) * nproc +
                sizeof(int) * (size_t)KMP_HW_LAST * 3;
  char *bytes = (char *)__kmp_allocate(size);
  retval = (kmp_topology_t *)bytes;
  if (nproc > 0) {
    retval->hw_threads = (kmp_hw_thread_t *)(bytes + sizeof(kmp_topology_t));
  } else {
    retval->hw_threads = nullptr;
  }
  retval->num_hw_threads = nproc;
  retval->depth = ndepth;
  int *arr =
      (int *)(bytes + sizeof(kmp_topology_t) + sizeof(kmp_hw_thread_t) * nproc);
  retval->types = (kmp_hw_t *)arr;
  retval->ratio = arr + (size_t)KMP_HW_LAST;
  retval->count = arr + 2 * (size_t)KMP_HW_LAST;
  retval->num_core_efficiencies = 0;
  retval->num_core_types = 0;
  retval->compact = 0;
  for (int i = 0; i < KMP_HW_MAX_NUM_CORE_TYPES; ++i)
    retval->core_types[i] = KMP_HW_CORE_TYPE_UNKNOWN;
  KMP_FOREACH_HW_TYPE(type) { retval->equivalent[type] = KMP_HW_UNKNOWN; }
  for (int i = 0; i < ndepth; ++i) {
    retval->types[i] = types[i];
    retval->equivalent[types[i]] = types[i];
  }
  return retval;
}

void kmp_topology_t::deallocate(kmp_topology_t *topology) {
  if (topology)
    __kmp_free(topology);
}

bool kmp_topology_t::check_ids() const {
  // Assume ids have been sorted
  if (num_hw_threads == 0)
    return true;
  for (int i = 1; i < num_hw_threads; ++i) {
    kmp_hw_thread_t &current_thread = hw_threads[i];
    kmp_hw_thread_t &previous_thread = hw_threads[i - 1];
    bool unique = false;
    for (int j = 0; j < depth; ++j) {
      if (previous_thread.ids[j] != current_thread.ids[j]) {
        unique = true;
        break;
      }
    }
    if (unique)
      continue;
    return false;
  }
  return true;
}

void kmp_topology_t::dump() const {
  printf("***********************\n");
  printf("*** __kmp_topology: ***\n");
  printf("***********************\n");
  printf("* depth: %d\n", depth);

  printf("* types: ");
  for (int i = 0; i < depth; ++i)
    printf("%15s ", __kmp_hw_get_keyword(types[i]));
  printf("\n");

  printf("* ratio: ");
  for (int i = 0; i < depth; ++i) {
    printf("%15d ", ratio[i]);
  }
  printf("\n");

  printf("* count: ");
  for (int i = 0; i < depth; ++i) {
    printf("%15d ", count[i]);
  }
  printf("\n");

  printf("* num_core_eff: %d\n", num_core_efficiencies);
  printf("* num_core_types: %d\n", num_core_types);
  printf("* core_types: ");
  for (int i = 0; i < num_core_types; ++i)
    printf("%3d ", core_types[i]);
  printf("\n");

  printf("* equivalent map:\n");
  KMP_FOREACH_HW_TYPE(i) {
    const char *key = __kmp_hw_get_keyword(i);
    const char *value = __kmp_hw_get_keyword(equivalent[i]);
    printf("%-15s -> %-15s\n", key, value);
  }

  printf("* uniform: %s\n", (is_uniform() ? "Yes" : "No"));

  printf("* num_hw_threads: %d\n", num_hw_threads);
  printf("* hw_threads:\n");
  for (int i = 0; i < num_hw_threads; ++i) {
    hw_threads[i].print();
  }
  printf("***********************\n");
}

void kmp_topology_t::print(const char *env_var) const {
  kmp_str_buf_t buf;
  int print_types_depth;
  __kmp_str_buf_init(&buf);
  kmp_hw_t print_types[KMP_HW_LAST + 2];

  // Num Available Threads
  if (num_hw_threads) {
    KMP_INFORM(AvailableOSProc, env_var, num_hw_threads);
  } else {
    KMP_INFORM(AvailableOSProc, env_var, __kmp_xproc);
  }

  // Uniform or not
  if (is_uniform()) {
    KMP_INFORM(Uniform, env_var);
  } else {
    KMP_INFORM(NonUniform, env_var);
  }

  // Equivalent types
  KMP_FOREACH_HW_TYPE(type) {
    kmp_hw_t eq_type = equivalent[type];
    if (eq_type != KMP_HW_UNKNOWN && eq_type != type) {
      KMP_INFORM(AffEqualTopologyTypes, env_var,
                 __kmp_hw_get_catalog_string(type),
                 __kmp_hw_get_catalog_string(eq_type));
    }
  }

  // Quick topology
  KMP_ASSERT(depth > 0 && depth <= (int)KMP_HW_LAST);
  // Create a print types array that always guarantees printing
  // the core and thread level
  print_types_depth = 0;
  for (int level = 0; level < depth; ++level)
    print_types[print_types_depth++] = types[level];
  if (equivalent[KMP_HW_CORE] != KMP_HW_CORE) {
    // Force in the core level for quick topology
    if (print_types[print_types_depth - 1] == KMP_HW_THREAD) {
      // Force core before thread e.g., 1 socket X 2 threads/socket
      // becomes 1 socket X 1 core/socket X 2 threads/socket
      print_types[print_types_depth - 1] = KMP_HW_CORE;
      print_types[print_types_depth++] = KMP_HW_THREAD;
    } else {
      print_types[print_types_depth++] = KMP_HW_CORE;
    }
  }
  // Always put threads at very end of quick topology
  if (equivalent[KMP_HW_THREAD] != KMP_HW_THREAD)
    print_types[print_types_depth++] = KMP_HW_THREAD;

  __kmp_str_buf_clear(&buf);
  kmp_hw_t numerator_type;
  kmp_hw_t denominator_type = KMP_HW_UNKNOWN;
  int core_level = get_level(KMP_HW_CORE);
  int ncores = get_count(core_level);

  for (int plevel = 0, level = 0; plevel < print_types_depth; ++plevel) {
    int c;
    bool plural;
    numerator_type = print_types[plevel];
    KMP_ASSERT_VALID_HW_TYPE(numerator_type);
    if (equivalent[numerator_type] != numerator_type)
      c = 1;
    else
      c = get_ratio(level++);
    plural = (c > 1);
    if (plevel == 0) {
      __kmp_str_buf_print(&buf, "%d %s", c,
                          __kmp_hw_get_catalog_string(numerator_type, plural));
    } else {
      __kmp_str_buf_print(&buf, " x %d %s/%s", c,
                          __kmp_hw_get_catalog_string(numerator_type, plural),
                          __kmp_hw_get_catalog_string(denominator_type));
    }
    denominator_type = numerator_type;
  }
  KMP_INFORM(TopologyGeneric, env_var, buf.str, ncores);

  // Hybrid topology information
  if (__kmp_is_hybrid_cpu()) {
    for (int i = 0; i < num_core_types; ++i) {
      kmp_hw_core_type_t core_type = core_types[i];
      kmp_hw_attr_t attr;
      attr.clear();
      attr.set_core_type(core_type);
      int ncores = get_ncores_with_attr(attr);
      if (ncores > 0) {
        KMP_INFORM(TopologyHybrid, env_var, ncores,
                   __kmp_hw_get_core_type_string(core_type));
        KMP_ASSERT(num_core_efficiencies <= KMP_HW_MAX_NUM_CORE_EFFS)
        for (int eff = 0; eff < num_core_efficiencies; ++eff) {
          attr.set_core_eff(eff);
          int ncores_with_eff = get_ncores_with_attr(attr);
          if (ncores_with_eff > 0) {
            KMP_INFORM(TopologyHybridCoreEff, env_var, ncores_with_eff, eff);
          }
        }
      }
    }
  }

  if (num_hw_threads <= 0) {
    __kmp_str_buf_free(&buf);
    return;
  }

  // Full OS proc to hardware thread map
  KMP_INFORM(OSProcToPhysicalThreadMap, env_var);
  for (int i = 0; i < num_hw_threads; i++) {
    __kmp_str_buf_clear(&buf);
    for (int level = 0; level < depth; ++level) {
      kmp_hw_t type = types[level];
      __kmp_str_buf_print(&buf, "%s ", __kmp_hw_get_catalog_string(type));
      __kmp_str_buf_print(&buf, "%d ", hw_threads[i].ids[level]);
    }
    if (__kmp_is_hybrid_cpu())
      __kmp_str_buf_print(
          &buf, "(%s)",
          __kmp_hw_get_core_type_string(hw_threads[i].attrs.get_core_type()));
    KMP_INFORM(OSProcMapToPack, env_var, hw_threads[i].os_id, buf.str);
  }

  __kmp_str_buf_free(&buf);
}

#if KMP_AFFINITY_SUPPORTED
void kmp_topology_t::set_granularity(kmp_affinity_t &affinity) const {
  const char *env_var = __kmp_get_affinity_env_var(affinity);
  // If requested hybrid CPU attributes for granularity (either OMP_PLACES or
  // KMP_AFFINITY), but none exist, then reset granularity and have below method
  // select a granularity and warn user.
  if (!__kmp_is_hybrid_cpu()) {
    if (affinity.core_attr_gran.valid) {
      // OMP_PLACES with cores:<attribute> but non-hybrid arch, use cores
      // instead
      KMP_AFF_WARNING(
          affinity, AffIgnoringNonHybrid, env_var,
          __kmp_hw_get_catalog_string(KMP_HW_CORE, /*plural=*/true));
      affinity.gran = KMP_HW_CORE;
      affinity.gran_levels = -1;
      affinity.core_attr_gran = KMP_AFFINITY_ATTRS_UNKNOWN;
      affinity.flags.core_types_gran = affinity.flags.core_effs_gran = 0;
    } else if (affinity.flags.core_types_gran ||
               affinity.flags.core_effs_gran) {
      // OMP_PLACES=core_types|core_effs but non-hybrid, use cores instead
      if (affinity.flags.omp_places) {
        KMP_AFF_WARNING(
            affinity, AffIgnoringNonHybrid, env_var,
            __kmp_hw_get_catalog_string(KMP_HW_CORE, /*plural=*/true));
      } else {
        // KMP_AFFINITY=granularity=core_type|core_eff,...
        KMP_AFF_WARNING(affinity, AffGranularityBad, env_var,
                        "Intel(R) Hybrid Technology core attribute",
                        __kmp_hw_get_catalog_string(KMP_HW_CORE));
      }
      affinity.gran = KMP_HW_CORE;
      affinity.gran_levels = -1;
      affinity.core_attr_gran = KMP_AFFINITY_ATTRS_UNKNOWN;
      affinity.flags.core_types_gran = affinity.flags.core_effs_gran = 0;
    }
  }
  // Set the number of affinity granularity levels
  if (affinity.gran_levels < 0) {
    kmp_hw_t gran_type = get_equivalent_type(affinity.gran);
    // Check if user's granularity request is valid
    if (gran_type == KMP_HW_UNKNOWN) {
      // First try core, then thread, then package
      kmp_hw_t gran_types[3] = {KMP_HW_CORE, KMP_HW_THREAD, KMP_HW_SOCKET};
      for (auto g : gran_types) {
        if (get_equivalent_type(g) != KMP_HW_UNKNOWN) {
          gran_type = g;
          break;
        }
      }
      KMP_ASSERT(gran_type != KMP_HW_UNKNOWN);
      // Warn user what granularity setting will be used instead
      KMP_AFF_WARNING(affinity, AffGranularityBad, env_var,
                      __kmp_hw_get_catalog_string(affinity.gran),
                      __kmp_hw_get_catalog_string(gran_type));
      affinity.gran = gran_type;
    }
#if KMP_GROUP_AFFINITY
    // If more than one processor group exists, and the level of
    // granularity specified by the user is too coarse, then the
    // granularity must be adjusted "down" to processor group affinity
    // because threads can only exist within one processor group.
    // For example, if a user sets granularity=socket and there are two
    // processor groups that cover a socket, then the runtime must
    // restrict the granularity down to the processor group level.
    if (__kmp_num_proc_groups > 1) {
      int gran_depth = get_level(gran_type);
      int proc_group_depth = get_level(KMP_HW_PROC_GROUP);
      if (gran_depth >= 0 && proc_group_depth >= 0 &&
          gran_depth < proc_group_depth) {
        KMP_AFF_WARNING(affinity, AffGranTooCoarseProcGroup, env_var,
                        __kmp_hw_get_catalog_string(affinity.gran));
        affinity.gran = gran_type = KMP_HW_PROC_GROUP;
      }
    }
#endif
    affinity.gran_levels = 0;
    for (int i = depth - 1; i >= 0 && get_type(i) != gran_type; --i)
      affinity.gran_levels++;
  }
}
#endif

void kmp_topology_t::canonicalize() {
#if KMP_GROUP_AFFINITY
  _insert_windows_proc_groups();
#endif
  _remove_radix1_layers();
  _gather_enumeration_information();
  _discover_uniformity();
  _set_sub_ids();
  _set_globals();
  _set_last_level_cache();

#if KMP_MIC_SUPPORTED
  // Manually Add L2 = Tile equivalence
  if (__kmp_mic_type == mic3) {
    if (get_level(KMP_HW_L2) != -1)
      set_equivalent_type(KMP_HW_TILE, KMP_HW_L2);
    else if (get_level(KMP_HW_TILE) != -1)
      set_equivalent_type(KMP_HW_L2, KMP_HW_TILE);
  }
#endif

  // Perform post canonicalization checking
  KMP_ASSERT(depth > 0);
  for (int level = 0; level < depth; ++level) {
    // All counts, ratios, and types must be valid
    KMP_ASSERT(count[level] > 0 && ratio[level] > 0);
    KMP_ASSERT_VALID_HW_TYPE(types[level]);
    // Detected types must point to themselves
    KMP_ASSERT(equivalent[types[level]] == types[level]);
  }
}

// Canonicalize an explicit packages X cores/pkg X threads/core topology
void kmp_topology_t::canonicalize(int npackages, int ncores_per_pkg,
                                  int nthreads_per_core, int ncores) {
  int ndepth = 3;
  depth = ndepth;
  KMP_FOREACH_HW_TYPE(i) { equivalent[i] = KMP_HW_UNKNOWN; }
  for (int level = 0; level < depth; ++level) {
    count[level] = 0;
    ratio[level] = 0;
  }
  count[0] = npackages;
  count[1] = ncores;
  count[2] = __kmp_xproc;
  ratio[0] = npackages;
  ratio[1] = ncores_per_pkg;
  ratio[2] = nthreads_per_core;
  equivalent[KMP_HW_SOCKET] = KMP_HW_SOCKET;
  equivalent[KMP_HW_CORE] = KMP_HW_CORE;
  equivalent[KMP_HW_THREAD] = KMP_HW_THREAD;
  types[0] = KMP_HW_SOCKET;
  types[1] = KMP_HW_CORE;
  types[2] = KMP_HW_THREAD;
  //__kmp_avail_proc = __kmp_xproc;
  _discover_uniformity();
}

// Represents running sub IDs for a single core attribute where
// attribute values have SIZE possibilities.
template <size_t SIZE, typename IndexFunc> struct kmp_sub_ids_t {
  int last_level; // last level in topology to consider for sub_ids
  int sub_id[SIZE]; // The sub ID for a given attribute value
  int prev_sub_id[KMP_HW_LAST];
  IndexFunc indexer;

public:
  kmp_sub_ids_t(int last_level) : last_level(last_level) {
    KMP_ASSERT(last_level < KMP_HW_LAST);
    for (size_t i = 0; i < SIZE; ++i)
      sub_id[i] = -1;
    for (size_t i = 0; i < KMP_HW_LAST; ++i)
      prev_sub_id[i] = -1;
  }
  void update(const kmp_hw_thread_t &hw_thread) {
    int idx = indexer(hw_thread);
    KMP_ASSERT(idx < (int)SIZE);
    for (int level = 0; level <= last_level; ++level) {
      if (hw_thread.sub_ids[level] != prev_sub_id[level]) {
        if (level < last_level)
          sub_id[idx] = -1;
        sub_id[idx]++;
        break;
      }
    }
    for (int level = 0; level <= last_level; ++level)
      prev_sub_id[level] = hw_thread.sub_ids[level];
  }
  int get_sub_id(const kmp_hw_thread_t &hw_thread) const {
    return sub_id[indexer(hw_thread)];
  }
};

#if KMP_AFFINITY_SUPPORTED
static kmp_str_buf_t *
__kmp_hw_get_catalog_core_string(const kmp_hw_attr_t &attr, kmp_str_buf_t *buf,
                                 bool plural) {
  __kmp_str_buf_init(buf);
  if (attr.is_core_type_valid())
    __kmp_str_buf_print(buf, "%s %s",
                        __kmp_hw_get_core_type_string(attr.get_core_type()),
                        __kmp_hw_get_catalog_string(KMP_HW_CORE, plural));
  else
    __kmp_str_buf_print(buf, "%s eff=%d",
                        __kmp_hw_get_catalog_string(KMP_HW_CORE, plural),
                        attr.get_core_eff());
  return buf;
}

bool kmp_topology_t::restrict_to_mask(const kmp_affin_mask_t *mask) {
  // Apply the filter
  bool affected;
  int new_index = 0;
  for (int i = 0; i < num_hw_threads; ++i) {
    int os_id = hw_threads[i].os_id;
    if (KMP_CPU_ISSET(os_id, mask)) {
      if (i != new_index)
        hw_threads[new_index] = hw_threads[i];
      new_index++;
    } else {
      KMP_CPU_CLR(os_id, __kmp_affin_fullMask);
      __kmp_avail_proc--;
    }
  }

  KMP_DEBUG_ASSERT(new_index <= num_hw_threads);
  affected = (num_hw_threads != new_index);
  num_hw_threads = new_index;

  // Post hardware subset canonicalization
  if (affected) {
    _gather_enumeration_information();
    _discover_uniformity();
    _set_globals();
    _set_last_level_cache();
#if KMP_OS_WINDOWS
    // Copy filtered full mask if topology has single processor group
    if (__kmp_num_proc_groups <= 1)
#endif
      __kmp_affin_origMask->copy(__kmp_affin_fullMask);
  }
  return affected;
}

// Apply the KMP_HW_SUBSET envirable to the topology
// Returns true if KMP_HW_SUBSET filtered any processors
// otherwise, returns false
bool kmp_topology_t::filter_hw_subset() {
  // If KMP_HW_SUBSET wasn't requested, then do nothing.
  if (!__kmp_hw_subset)
    return false;

  // First, sort the KMP_HW_SUBSET items by the machine topology
  __kmp_hw_subset->sort();

  // Check to see if KMP_HW_SUBSET is a valid subset of the detected topology
  bool using_core_types = false;
  bool using_core_effs = false;
  int hw_subset_depth = __kmp_hw_subset->get_depth();
  kmp_hw_t specified[KMP_HW_LAST];
  int *topology_levels = (int *)KMP_ALLOCA(sizeof(int) * hw_subset_depth);
  KMP_ASSERT(hw_subset_depth > 0);
  KMP_FOREACH_HW_TYPE(i) { specified[i] = KMP_HW_UNKNOWN; }
  int core_level = get_level(KMP_HW_CORE);
  for (int i = 0; i < hw_subset_depth; ++i) {
    int max_count;
    const kmp_hw_subset_t::item_t &item = __kmp_hw_subset->at(i);
    int num = item.num[0];
    int offset = item.offset[0];
    kmp_hw_t type = item.type;
    kmp_hw_t equivalent_type = equivalent[type];
    int level = get_level(type);
    topology_levels[i] = level;

    // Check to see if current layer is in detected machine topology
    if (equivalent_type != KMP_HW_UNKNOWN) {
      __kmp_hw_subset->at(i).type = equivalent_type;
    } else {
      KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetNotExistGeneric,
                      __kmp_hw_get_catalog_string(type));
      return false;
    }

    // Check to see if current layer has already been
    // specified either directly or through an equivalent type
    if (specified[equivalent_type] != KMP_HW_UNKNOWN) {
      KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetEqvLayers,
                      __kmp_hw_get_catalog_string(type),
                      __kmp_hw_get_catalog_string(specified[equivalent_type]));
      return false;
    }
    specified[equivalent_type] = type;

    // Check to see if each layer's num & offset parameters are valid
    max_count = get_ratio(level);
    if (max_count < 0 ||
        (num != kmp_hw_subset_t::USE_ALL && num + offset > max_count)) {
      bool plural = (num > 1);
      KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetManyGeneric,
                      __kmp_hw_get_catalog_string(type, plural));
      return false;
    }

    // Check to see if core attributes are consistent
    if (core_level == level) {
      // Determine which core attributes are specified
      for (int j = 0; j < item.num_attrs; ++j) {
        if (item.attr[j].is_core_type_valid())
          using_core_types = true;
        if (item.attr[j].is_core_eff_valid())
          using_core_effs = true;
      }

      // Check if using a single core attribute on non-hybrid arch.
      // Do not ignore all of KMP_HW_SUBSET, just ignore the attribute.
      //
      // Check if using multiple core attributes on non-hyrbid arch.
      // Ignore all of KMP_HW_SUBSET if this is the case.
      if ((using_core_effs || using_core_types) && !__kmp_is_hybrid_cpu()) {
        if (item.num_attrs == 1) {
          if (using_core_effs) {
            KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetIgnoringAttr,
                            "efficiency");
          } else {
            KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetIgnoringAttr,
                            "core_type");
          }
          using_core_effs = false;
          using_core_types = false;
        } else {
          KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetAttrsNonHybrid);
          return false;
        }
      }

      // Check if using both core types and core efficiencies together
      if (using_core_types && using_core_effs) {
        KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetIncompat, "core_type",
                        "efficiency");
        return false;
      }

      // Check that core efficiency values are valid
      if (using_core_effs) {
        for (int j = 0; j < item.num_attrs; ++j) {
          if (item.attr[j].is_core_eff_valid()) {
            int core_eff = item.attr[j].get_core_eff();
            if (core_eff < 0 || core_eff >= num_core_efficiencies) {
              kmp_str_buf_t buf;
              __kmp_str_buf_init(&buf);
              __kmp_str_buf_print(&buf, "%d", item.attr[j].get_core_eff());
              __kmp_msg(kmp_ms_warning,
                        KMP_MSG(AffHWSubsetAttrInvalid, "efficiency", buf.str),
                        KMP_HNT(ValidValuesRange, 0, num_core_efficiencies - 1),
                        __kmp_msg_null);
              __kmp_str_buf_free(&buf);
              return false;
            }
          }
        }
      }

      // Check that the number of requested cores with attributes is valid
      if (using_core_types || using_core_effs) {
        for (int j = 0; j < item.num_attrs; ++j) {
          int num = item.num[j];
          int offset = item.offset[j];
          int level_above = core_level - 1;
          if (level_above >= 0) {
            max_count = get_ncores_with_attr_per(item.attr[j], level_above);
            if (max_count <= 0 ||
                (num != kmp_hw_subset_t::USE_ALL && num + offset > max_count)) {
              kmp_str_buf_t buf;
              __kmp_hw_get_catalog_core_string(item.attr[j], &buf, num > 0);
              KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetManyGeneric, buf.str);
              __kmp_str_buf_free(&buf);
              return false;
            }
          }
        }
      }

      if ((using_core_types || using_core_effs) && item.num_attrs > 1) {
        for (int j = 0; j < item.num_attrs; ++j) {
          // Ambiguous use of specific core attribute + generic core
          // e.g., 4c & 3c:intel_core or 4c & 3c:eff1
          if (!item.attr[j]) {
            kmp_hw_attr_t other_attr;
            for (int k = 0; k < item.num_attrs; ++k) {
              if (item.attr[k] != item.attr[j]) {
                other_attr = item.attr[k];
                break;
              }
            }
            kmp_str_buf_t buf;
            __kmp_hw_get_catalog_core_string(other_attr, &buf, item.num[j] > 0);
            KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetIncompat,
                            __kmp_hw_get_catalog_string(KMP_HW_CORE), buf.str);
            __kmp_str_buf_free(&buf);
            return false;
          }
          // Allow specifying a specific core type or core eff exactly once
          for (int k = 0; k < j; ++k) {
            if (!item.attr[j] || !item.attr[k])
              continue;
            if (item.attr[k] == item.attr[j]) {
              kmp_str_buf_t buf;
              __kmp_hw_get_catalog_core_string(item.attr[j], &buf,
                                               item.num[j] > 0);
              KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetAttrRepeat, buf.str);
              __kmp_str_buf_free(&buf);
              return false;
            }
          }
        }
      }
    }
  }

  struct core_type_indexer {
    int operator()(const kmp_hw_thread_t &t) const {
      switch (t.attrs.get_core_type()) {
      case KMP_HW_CORE_TYPE_UNKNOWN:
      case KMP_HW_MAX_NUM_CORE_TYPES:
        return 0;
#if KMP_ARCH_X86 || KMP_ARCH_X86_64
      case KMP_HW_CORE_TYPE_ATOM:
        return 1;
      case KMP_HW_CORE_TYPE_CORE:
        return 2;
#endif
      }
      KMP_ASSERT2(false, "Unhandled kmp_hw_thread_t enumeration");
      KMP_BUILTIN_UNREACHABLE;
    }
  };
  struct core_eff_indexer {
    int operator()(const kmp_hw_thread_t &t) const {
      return t.attrs.get_core_eff();
    }
  };

  kmp_sub_ids_t<KMP_HW_MAX_NUM_CORE_TYPES, core_type_indexer> core_type_sub_ids(
      core_level);
  kmp_sub_ids_t<KMP_HW_MAX_NUM_CORE_EFFS, core_eff_indexer> core_eff_sub_ids(
      core_level);

  // Determine which hardware threads should be filtered.
  int num_filtered = 0;
  kmp_affin_mask_t *filtered_mask;
  KMP_CPU_ALLOC(filtered_mask);
  KMP_CPU_COPY(filtered_mask, __kmp_affin_fullMask);
  for (int i = 0; i < num_hw_threads; ++i) {
    kmp_hw_thread_t &hw_thread = hw_threads[i];
    // Update type_sub_id
    if (using_core_types)
      core_type_sub_ids.update(hw_thread);
    if (using_core_effs)
      core_eff_sub_ids.update(hw_thread);

    // Check to see if this hardware thread should be filtered
    bool should_be_filtered = false;
    for (int hw_subset_index = 0; hw_subset_index < hw_subset_depth;
         ++hw_subset_index) {
      const auto &hw_subset_item = __kmp_hw_subset->at(hw_subset_index);
      int level = topology_levels[hw_subset_index];
      if (level == -1)
        continue;
      if ((using_core_effs || using_core_types) && level == core_level) {
        // Look for the core attribute in KMP_HW_SUBSET which corresponds
        // to this hardware thread's core attribute. Use this num,offset plus
        // the running sub_id for the particular core attribute of this hardware
        // thread to determine if the hardware thread should be filtered or not.
        int attr_idx;
        kmp_hw_core_type_t core_type = hw_thread.attrs.get_core_type();
        int core_eff = hw_thread.attrs.get_core_eff();
        for (attr_idx = 0; attr_idx < hw_subset_item.num_attrs; ++attr_idx) {
          if (using_core_types &&
              hw_subset_item.attr[attr_idx].get_core_type() == core_type)
            break;
          if (using_core_effs &&
              hw_subset_item.attr[attr_idx].get_core_eff() == core_eff)
            break;
        }
        // This core attribute isn't in the KMP_HW_SUBSET so always filter it.
        if (attr_idx == hw_subset_item.num_attrs) {
          should_be_filtered = true;
          break;
        }
        int sub_id;
        int num = hw_subset_item.num[attr_idx];
        int offset = hw_subset_item.offset[attr_idx];
        if (using_core_types)
          sub_id = core_type_sub_ids.get_sub_id(hw_thread);
        else
          sub_id = core_eff_sub_ids.get_sub_id(hw_thread);
        if (sub_id < offset ||
            (num != kmp_hw_subset_t::USE_ALL && sub_id >= offset + num)) {
          should_be_filtered = true;
          break;
        }
      } else {
        int num = hw_subset_item.num[0];
        int offset = hw_subset_item.offset[0];
        if (hw_thread.sub_ids[level] < offset ||
            (num != kmp_hw_subset_t::USE_ALL &&
             hw_thread.sub_ids[level] >= offset + num)) {
          should_be_filtered = true;
          break;
        }
      }
    }
    // Collect filtering information
    if (should_be_filtered) {
      KMP_CPU_CLR(hw_thread.os_id, filtered_mask);
      num_filtered++;
    }
  }

  // One last check that we shouldn't allow filtering entire machine
  if (num_filtered == num_hw_threads) {
    KMP_AFF_WARNING(__kmp_affinity, AffHWSubsetAllFiltered);
    return false;
  }

  // Apply the filter
  restrict_to_mask(filtered_mask);
  return true;
}

bool kmp_topology_t::is_close(int hwt1, int hwt2,
                              const kmp_affinity_t &stgs) const {
  int hw_level = stgs.gran_levels;
  if (hw_level >= depth)
    return true;
  bool retval = true;
  const kmp_hw_thread_t &t1 = hw_threads[hwt1];
  const kmp_hw_thread_t &t2 = hw_threads[hwt2];
  if (stgs.flags.core_types_gran)
    return t1.attrs.get_core_type() == t2.attrs.get_core_type();
  if (stgs.flags.core_effs_gran)
    return t1.attrs.get_core_eff() == t2.attrs.get_core_eff();
  for (int i = 0; i < (depth - hw_level); ++i) {
    if (t1.ids[i] != t2.ids[i])
      return false;
  }
  return retval;
}

////////////////////////////////////////////////////////////////////////////////

bool KMPAffinity::picked_api = false;

void *KMPAffinity::Mask::operator new(size_t n) { return __kmp_allocate(n); }
void *KMPAffinity::Mask::operator new[](size_t n) { return __kmp_allocate(n); }
void KMPAffinity::Mask::operator delete(void *p) { __kmp_free(p); }
void KMPAffinity::Mask::operator delete[](void *p) { __kmp_free(p); }
void *KMPAffinity::operator new(size_t n) { return __kmp_allocate(n); }
void KMPAffinity::operator delete(void *p) { __kmp_free(p); }

void KMPAffinity::pick_api() {
  KMPAffinity *affinity_dispatch;
  if (picked_api)
    return;
#if KMP_USE_HWLOC
  // Only use Hwloc if affinity isn't explicitly disabled and
  // user requests Hwloc topology method
  if (__kmp_affinity_top_method == affinity_top_method_hwloc &&
      __kmp_affinity.type != affinity_disabled) {
    affinity_dispatch = new KMPHwlocAffinity();
  } else
#endif
  {
    affinity_dispatch = new KMPNativeAffinity();
  }
  __kmp_affinity_dispatch = affinity_dispatch;
  picked_api = true;
}

void KMPAffinity::destroy_api() {
  if (__kmp_affinity_dispatch != NULL) {
    delete __kmp_affinity_dispatch;
    __kmp_affinity_dispatch = NULL;
    picked_api = false;
  }
}

#define KMP_ADVANCE_SCAN(scan)                                                 \
  while (*scan != '\0') {                                                      \
    scan++;                                                                    \
  }

// Print the affinity mask to the character array in a pretty format.
// The format is a comma separated list of non-negative integers or integer
// ranges: e.g., 1,2,3-5,7,9-15
// The format can also be the string "{<empty>}" if no bits are set in mask
char *__kmp_affinity_print_mask(char *buf, int buf_len,
                                kmp_affin_mask_t *mask) {
  int start = 0, finish = 0, previous = 0;
  bool first_range;
  KMP_ASSERT(buf);
  KMP_ASSERT(buf_len >= 40);
  KMP_ASSERT(mask);
  char *scan = buf;
  char *end = buf + buf_len - 1;

  // Check for empty set.
  if (mask->begin() == mask->end()) {
    KMP_SNPRINTF(scan, end - scan + 1, "{<empty>}");
    KMP_ADVANCE_SCAN(scan);
    KMP_ASSERT(scan <= end);
    return buf;
  }

  first_range = true;
  start = mask->begin();
  while (1) {
    // Find next range
    // [start, previous] is inclusive range of contiguous bits in mask
    for (finish = mask->next(start), previous = start;
         finish == previous + 1 && finish != mask->end();
         finish = mask->next(finish)) {
      previous = finish;
    }

    // The first range does not need a comma printed before it, but the rest
    // of the ranges do need a comma beforehand
    if (!first_range) {
      KMP_SNPRINTF(scan, end - scan + 1, "%s", ",");
      KMP_ADVANCE_SCAN(scan);
    } else {
      first_range = false;
    }
    // Range with three or more contiguous bits in the affinity mask
    if (previous - start > 1) {
      KMP_SNPRINTF(scan, end - scan + 1, "%u-%u", start, previous);
    } else {
      // Range with one or two contiguous bits in the affinity mask
      KMP_SNPRINTF(scan, end - scan + 1, "%u", start);
      KMP_ADVANCE_SCAN(scan);
      if (previous - start > 0) {
        KMP_SNPRINTF(scan, end - scan + 1, ",%u", previous);
      }
    }
    KMP_ADVANCE_SCAN(scan);
    // Start over with new start point
    start = finish;
    if (start == mask->end())
      break;
    // Check for overflow
    if (end - scan < 2)
      break;
  }

  // Check for overflow
  KMP_ASSERT(scan <= end);
  return buf;
}
#undef KMP_ADVANCE_SCAN

// Print the affinity mask to the string buffer object in a pretty format
// The format is a comma separated list of non-negative integers or integer
// ranges: e.g., 1,2,3-5,7,9-15
// The format can also be the string "{<empty>}" if no bits are set in mask
kmp_str_buf_t *__kmp_affinity_str_buf_mask(kmp_str_buf_t *buf,
                                           kmp_affin_mask_t *mask) {
  int start = 0, finish = 0, previous = 0;
  bool first_range;
  KMP_ASSERT(buf);
  KMP_ASSERT(mask);

  __kmp_str_buf_clear(buf);

  // Check for empty set.
  if (mask->begin() == mask->end()) {
    __kmp_str_buf_print(buf, "%s", "{<empty>}");
    return buf;
  }

  first_range = true;
  start = mask->begin();
  while (1) {
    // Find next range
    // [start, previous] is inclusive range of contiguous bits in mask
    for (finish = mask->next(start), previous = start;
         finish == previous + 1 && finish != mask->end();
         finish = mask->next(finish)) {
      previous = finish;
    }

    // The first range does not need a comma printed before it, but the rest
    // of the ranges do need a comma beforehand
    if (!first_range) {
      __kmp_str_buf_print(buf, "%s", ",");
    } else {
      first_range = false;
    }
    // Range with three or more contiguous bits in the affinity mask
    if (previous - start > 1) {
      __kmp_str_buf_print(buf, "%u-%u", start, previous);
    } else {
      // Range with one or two contiguous bits in the affinity mask
      __kmp_str_buf_print(buf, "%u", start);
      if (previous - start > 0) {
        __kmp_str_buf_print(buf, ",%u", previous);
      }
    }
    // Start over with new start point
    start = finish;
    if (start == mask->end())
      break;
  }
  return buf;
}

// Return (possibly empty) affinity mask representing the offline CPUs
// Caller must free the mask
kmp_affin_mask_t *__kmp_affinity_get_offline_cpus() {
  kmp_affin_mask_t *offline;
  KMP_CPU_ALLOC(offline);
  KMP_CPU_ZERO(offline);
#if KMP_OS_LINUX
  int n, begin_cpu, end_cpu;
  kmp_safe_raii_file_t offline_file;
  auto skip_ws = [](FILE *f) {
    int c;
    do {
      c = fgetc(f);
    } while (isspace(c));
    if (c != EOF)
      ungetc(c, f);
  };
  // File contains CSV of integer ranges representing the offline CPUs
  // e.g., 1,2,4-7,9,11-15
  int status = offline_file.try_open("/sys/devices/system/cpu/offline", "r");
  if (status != 0)
    return offline;
  while (!feof(offline_file)) {
    skip_ws(offline_file);
    n = fscanf(offline_file, "%d", &begin_cpu);
    if (n != 1)
      break;
    skip_ws(offline_file);
    int c = fgetc(offline_file);
    if (c == EOF || c == ',') {
      // Just single CPU
      end_cpu = begin_cpu;
    } else if (c == '-') {
      // Range of CPUs
      skip_ws(offline_file);
      n = fscanf(offline_file, "%d", &end_cpu);
      if (n != 1)
        break;
      skip_ws(offline_file);
      c = fgetc(offline_file); // skip ','
    } else {
      // Syntax problem
      break;
    }
    // Ensure a valid range of CPUs
    if (begin_cpu < 0 || begin_cpu >= __kmp_xproc || end_cpu < 0 ||
        end_cpu >= __kmp_xproc || begin_cpu > end_cpu) {
      continue;
    }
    // Insert [begin_cpu, end_cpu] into offline mask
    for (int cpu = begin_cpu; cpu <= end_cpu; ++cpu) {
      KMP_CPU_SET(cpu, offline);
    }
  }
#endif
  return offline;
}

// Return the number of available procs
int __kmp_affinity_entire_machine_mask(kmp_affin_mask_t *mask) {
  int avail_proc = 0;
  KMP_CPU_ZERO(mask);

#if KMP_GROUP_AFFINITY

  if (__kmp_num_proc_groups > 1) {
    int group;
    KMP_DEBUG_ASSERT(__kmp_GetActiveProcessorCount != NULL);
    for (group = 0; group < __kmp_num_proc_groups; group++) {
      int i;
      int num = __kmp_GetActiveProcessorCount(group);
      for (i = 0; i < num; i++) {
        KMP_CPU_SET(i + group * (CHAR_BIT * sizeof(DWORD_PTR)), mask);
        avail_proc++;
      }
    }
  } else

#endif /* KMP_GROUP_AFFINITY */

  {
    int proc;
    kmp_affin_mask_t *offline_cpus = __kmp_affinity_get_offline_cpus();
    for (proc = 0; proc < __kmp_xproc; proc++) {
      // Skip offline CPUs
      if (KMP_CPU_ISSET(proc, offline_cpus))
        continue;
      KMP_CPU_SET(proc, mask);
      avail_proc++;
    }
    KMP_CPU_FREE(offline_cpus);
  }

  return avail_proc;
}

// All of the __kmp_affinity_create_*_map() routines should allocate the
// internal topology object and set the layer ids for it.  Each routine
// returns a boolean on whether it was successful at doing so.
kmp_affin_mask_t *__kmp_affin_fullMask = NULL;
// Original mask is a subset of full mask in multiple processor groups topology
kmp_affin_mask_t *__kmp_affin_origMask = NULL;

#if KMP_USE_HWLOC
static inline bool __kmp_hwloc_is_cache_type(hwloc_obj_t obj) {
#if HWLOC_API_VERSION >= 0x00020000
  return hwloc_obj_type_is_cache(obj->type);
#else
  return obj->type == HWLOC_OBJ_CACHE;
#endif
}

// Returns KMP_HW_* type derived from HWLOC_* type
static inline kmp_hw_t __kmp_hwloc_type_2_topology_type(hwloc_obj_t obj) {

  if (__kmp_hwloc_is_cache_type(obj)) {
    if (obj->attr->cache.type == HWLOC_OBJ_CACHE_INSTRUCTION)
      return KMP_HW_UNKNOWN;
    switch (obj->attr->cache.depth) {
    case 1:
      return KMP_HW_L1;
    case 2:
#if KMP_MIC_SUPPORTED
      if (__kmp_mic_type == mic3) {
        return KMP_HW_TILE;
      }
#endif
      return KMP_HW_L2;
    case 3:
      return KMP_HW_L3;
    }
    return KMP_HW_UNKNOWN;
  }

  switch (obj->type) {
  case HWLOC_OBJ_PACKAGE:
    return KMP_HW_SOCKET;
  case HWLOC_OBJ_NUMANODE:
    return KMP_HW_NUMA;
  case HWLOC_OBJ_CORE:
    return KMP_HW_CORE;
  case HWLOC_OBJ_PU:
    return KMP_HW_THREAD;
  case HWLOC_OBJ_GROUP:
#if HWLOC_API_VERSION >= 0x00020000
    if (obj->attr->group.kind == HWLOC_GROUP_KIND_INTEL_DIE)
      return KMP_HW_DIE;
    else if (obj->attr->group.kind == HWLOC_GROUP_KIND_INTEL_TILE)
      return KMP_HW_TILE;
    else if (obj->attr->group.kind == HWLOC_GROUP_KIND_INTEL_MODULE)
      return KMP_HW_MODULE;
    else if (obj->attr->group.kind == HWLOC_GROUP_KIND_WINDOWS_PROCESSOR_GROUP)
      return KMP_HW_PROC_GROUP;
#endif
    return KMP_HW_UNKNOWN;
#if HWLOC_API_VERSION >= 0x00020100
  case HWLOC_OBJ_DIE:
    return KMP_HW_DIE;
#endif
  }
  return KMP_HW_UNKNOWN;
}

// Returns the number of objects of type 'type' below 'obj' within the topology
// tree structure. e.g., if obj is a HWLOC_OBJ_PACKAGE object, and type is
// HWLOC_OBJ_PU, then this will return the number of PU's under the SOCKET
// object.
static int __kmp_hwloc_get_nobjs_under_obj(hwloc_obj_t obj,
                                           hwloc_obj_type_t type) {
  int retval = 0;
  hwloc_obj_t first;
  for (first = hwloc_get_obj_below_by_type(__kmp_hwloc_topology, obj->type,
                                           obj->logical_index, type, 0);
       first != NULL && hwloc_get_ancestor_obj_by_type(__kmp_hwloc_topology,
                                                       obj->type, first) == obj;
       first = hwloc_get_next_obj_by_type(__kmp_hwloc_topology, first->type,
                                          first)) {
    ++retval;
  }
  return retval;
}

// This gets the sub_id for a lower object under a higher object in the
// topology tree
static int __kmp_hwloc_get_sub_id(hwloc_topology_t t, hwloc_obj_t higher,
                                  hwloc_obj_t lower) {
  hwloc_obj_t obj;
  hwloc_obj_type_t ltype = lower->type;
  int lindex = lower->logical_index - 1;
  int sub_id = 0;
  // Get the previous lower object
  obj = hwloc_get_obj_by_type(t, ltype, lindex);
  while (obj && lindex >= 0 &&
         hwloc_bitmap_isincluded(obj->cpuset, higher->cpuset)) {
    if (obj->userdata) {
      sub_id = (int)(RCAST(kmp_intptr_t, obj->userdata));
      break;
    }
    sub_id++;
    lindex--;
    obj = hwloc_get_obj_by_type(t, ltype, lindex);
  }
  // store sub_id + 1 so that 0 is differed from NULL
  lower->userdata = RCAST(void *, sub_id + 1);
  return sub_id;
}

static bool __kmp_affinity_create_hwloc_map(kmp_i18n_id_t *const msg_id) {
  kmp_hw_t type;
  int hw_thread_index, sub_id;
  int depth;
  hwloc_obj_t pu, obj, root, prev;
  kmp_hw_t types[KMP_HW_LAST];
  hwloc_obj_type_t hwloc_types[KMP_HW_LAST];

  hwloc_topology_t tp = __kmp_hwloc_topology;
  *msg_id = kmp_i18n_null;
  if (__kmp_affinity.flags.verbose) {
    KMP_INFORM(AffUsingHwloc, "KMP_AFFINITY");
  }

  if (!KMP_AFFINITY_CAPABLE()) {
    // Hack to try and infer the machine topology using only the data
    // available from hwloc on the current thread, and __kmp_xproc.
    KMP_ASSERT(__kmp_affinity.type == affinity_none);
    // hwloc only guarantees existance of PU object, so check PACKAGE and CORE
    hwloc_obj_t o = hwloc_get_obj_by_type(tp, HWLOC_OBJ_PACKAGE, 0);
    if (o != NULL)
      nCoresPerPkg = __kmp_hwloc_get_nobjs_under_obj(o, HWLOC_OBJ_CORE);
    else
      nCoresPerPkg = 1; // no PACKAGE found
    o = hwloc_get_obj_by_type(tp, HWLOC_OBJ_CORE, 0);
    if (o != NULL)
      __kmp_nThreadsPerCore = __kmp_hwloc_get_nobjs_under_obj(o, HWLOC_OBJ_PU);
    else
      __kmp_nThreadsPerCore = 1; // no CORE found
    __kmp_ncores = __kmp_xproc / __kmp_nThreadsPerCore;
    if (nCoresPerPkg == 0)
      nCoresPerPkg = 1; // to prevent possible division by 0
    nPackages = (__kmp_xproc + nCoresPerPkg - 1) / nCoresPerPkg;
    return true;
  }

#if HWLOC_API_VERSION >= 0x00020400
  // Handle multiple types of cores if they exist on the system
  int nr_cpu_kinds = hwloc_cpukinds_get_nr(tp, 0);

  typedef struct kmp_hwloc_cpukinds_info_t {
    int efficiency;
    kmp_hw_core_type_t core_type;
    hwloc_bitmap_t mask;
  } kmp_hwloc_cpukinds_info_t;
  kmp_hwloc_cpukinds_info_t *cpukinds = nullptr;

  if (nr_cpu_kinds > 0) {
    unsigned nr_infos;
    struct hwloc_info_s *infos;
    cpukinds = (kmp_hwloc_cpukinds_info_t *)__kmp_allocate(
        sizeof(kmp_hwloc_cpukinds_info_t) * nr_cpu_kinds);
    for (unsigned idx = 0; idx < (unsigned)nr_cpu_kinds; ++idx) {
      cpukinds[idx].efficiency = -1;
      cpukinds[idx].core_type = KMP_HW_CORE_TYPE_UNKNOWN;
      cpukinds[idx].mask = hwloc_bitmap_alloc();
      if (hwloc_cpukinds_get_info(tp, idx, cpukinds[idx].mask,
                                  &cpukinds[idx].efficiency, &nr_infos, &infos,
                                  0) == 0) {
        for (unsigned i = 0; i < nr_infos; ++i) {
          if (__kmp_str_match("CoreType", 8, infos[i].name)) {
#if KMP_ARCH_X86 || KMP_ARCH_X86_64
            if (__kmp_str_match("IntelAtom", 9, infos[i].value)) {
              cpukinds[idx].core_type = KMP_HW_CORE_TYPE_ATOM;
              break;
            } else if (__kmp_str_match("IntelCore", 9, infos[i].value)) {
              cpukinds[idx].core_type = KMP_HW_CORE_TYPE_CORE;
              break;
            }
#endif
          }
        }
      }
    }
  }
#endif

  root = hwloc_get_root_obj(tp);

  // Figure out the depth and types in the topology
  depth = 0;
  pu = hwloc_get_pu_obj_by_os_index(tp, __kmp_affin_fullMask->begin());
  KMP_ASSERT(pu);
  obj = pu;
  types[depth] = KMP_HW_THREAD;
  hwloc_types[depth] = obj->type;
  depth++;
  while (obj != root && obj != NULL) {
    obj = obj->parent;
#if HWLOC_API_VERSION >= 0x00020000
    if (obj->memory_arity) {
      hwloc_obj_t memory;
      for (memory = obj->memory_first_child; memory;
           memory = hwloc_get_next_child(tp, obj, memory)) {
        if (memory->type == HWLOC_OBJ_NUMANODE)
          break;
      }
      if (memory && memory->type == HWLOC_OBJ_NUMANODE) {
        types[depth] = KMP_HW_NUMA;
        hwloc_types[depth] = memory->type;
        depth++;
      }
    }
#endif
    type = __kmp_hwloc_type_2_topology_type(obj);
    if (type != KMP_HW_UNKNOWN) {
      types[depth] = type;
      hwloc_types[depth] = obj->type;
      depth++;
    }
  }
  KMP_ASSERT(depth > 0);

  // Get the order for the types correct
  for (int i = 0, j = depth - 1; i < j; ++i, --j) {
    hwloc_obj_type_t hwloc_temp = hwloc_types[i];
    kmp_hw_t temp = types[i];
    types[i] = types[j];
    types[j] = temp;
    hwloc_types[i] = hwloc_types[j];
    hwloc_types[j] = hwloc_temp;
  }

  // Allocate the data structure to be returned.
  __kmp_topology = kmp_topology_t::allocate(__kmp_avail_proc, depth, types);

  hw_thread_index = 0;
  pu = NULL;
  while ((pu = hwloc_get_next_obj_by_type(tp, HWLOC_OBJ_PU, pu))) {
    int index = depth - 1;
    bool included = KMP_CPU_ISSET(pu->os_index, __kmp_affin_fullMask);
    kmp_hw_thread_t &hw_thread = __kmp_topology->at(hw_thread_index);
    if (included) {
      hw_thread.clear();
      hw_thread.ids[index] = pu->logical_index;
      hw_thread.os_id = pu->os_index;
      // If multiple core types, then set that attribute for the hardware thread
#if HWLOC_API_VERSION >= 0x00020400
      if (cpukinds) {
        int cpukind_index = -1;
        for (int i = 0; i < nr_cpu_kinds; ++i) {
          if (hwloc_bitmap_isset(cpukinds[i].mask, hw_thread.os_id)) {
            cpukind_index = i;
            break;
          }
        }
        if (cpukind_index >= 0) {
          hw_thread.attrs.set_core_type(cpukinds[cpukind_index].core_type);
          hw_thread.attrs.set_core_eff(cpukinds[cpukind_index].efficiency);
        }
      }
#endif
      index--;
    }
    obj = pu;
    prev = obj;
    while (obj != root && obj != NULL) {
      obj = obj->parent;
#if HWLOC_API_VERSION >= 0x00020000
      // NUMA Nodes are handled differently since they are not within the
      // parent/child structure anymore.  They are separate children
      // of obj (memory_first_child points to first memory child)
      if (obj->memory_arity) {
        hwloc_obj_t memory;
        for (memory = obj->memory_first_child; memory;
             memory = hwloc_get_next_child(tp, obj, memory)) {
          if (memory->type == HWLOC_OBJ_NUMANODE)
            break;
        }
        if (memory && memory->type == HWLOC_OBJ_NUMANODE) {
          sub_id = __kmp_hwloc_get_sub_id(tp, memory, prev);
          if (included) {
            hw_thread.ids[index] = memory->logical_index;
            hw_thread.ids[index + 1] = sub_id;
            index--;
          }
          prev = memory;
        }
        prev = obj;
      }
#endif
      type = __kmp_hwloc_type_2_topology_type(obj);
      if (type != KMP_HW_UNKNOWN) {
        sub_id = __kmp_hwloc_get_sub_id(tp, obj, prev);
        if (included) {
          hw_thread.ids[index] = obj->logical_index;
          hw_thread.ids[index + 1] = sub_id;
          index--;
        }
        prev = obj;
      }
    }
    if (included)
      hw_thread_index++;
  }

#if HWLOC_API_VERSION >= 0x00020400
  // Free the core types information
  if (cpukinds) {
    for (int idx = 0; idx < nr_cpu_kinds; ++idx)
      hwloc_bitmap_free(cpukinds[idx].mask);
    __kmp_free(cpukinds);
  }
#endif
  __kmp_topology->sort_ids();
  return true;
}
#endif // KMP_USE_HWLOC

// If we don't know how to retrieve the machine's processor topology, or
// encounter an error in doing so, this routine is called to form a "flat"
// mapping of os thread id's <-> processor id's.
static bool __kmp_affinity_create_flat_map(kmp_i18n_id_t *const msg_id) {
  *msg_id = kmp_i18n_null;
  int depth = 3;
  kmp_hw_t types[] = {KMP_HW_SOCKET, KMP_HW_CORE, KMP_HW_THREAD};

  if (__kmp_affinity.flags.verbose) {
    KMP_INFORM(UsingFlatOS, "KMP_AFFINITY");
  }

  // Even if __kmp_affinity.type == affinity_none, this routine might still
  // be called to set __kmp_ncores, as well as
  // __kmp_nThreadsPerCore, nCoresPerPkg, & nPackages.
  if (!KMP_AFFINITY_CAPABLE()) {
    KMP_ASSERT(__kmp_affinity.type == affinity_none);
    __kmp_ncores = nPackages = __kmp_xproc;
    __kmp_nThreadsPerCore = nCoresPerPkg = 1;
    return true;
  }

  // When affinity is off, this routine will still be called to set
  // __kmp_ncores, as well as __kmp_nThreadsPerCore, nCoresPerPkg, & nPackages.
  // Make sure all these vars are set correctly, and return now if affinity is
  // not enabled.
  __kmp_ncores = nPackages = __kmp_avail_proc;
  __kmp_nThreadsPerCore = nCoresPerPkg = 1;

  // Construct the data structure to be returned.
  __kmp_topology = kmp_topology_t::allocate(__kmp_avail_proc, depth, types);
  int avail_ct = 0;
  int i;
  KMP_CPU_SET_ITERATE(i, __kmp_affin_fullMask) {
    // Skip this proc if it is not included in the machine model.
    if (!KMP_CPU_ISSET(i, __kmp_affin_fullMask)) {
      continue;
    }
    kmp_hw_thread_t &hw_thread = __kmp_topology->at(avail_ct);
    hw_thread.clear();
    hw_thread.os_id = i;
    hw_thread.ids[0] = i;
    hw_thread.ids[1] = 0;
    hw_thread.ids[2] = 0;
    avail_ct++;
  }
  if (__kmp_affinity.flags.verbose) {
    KMP_INFORM(OSProcToPackage, "KMP_AFFINITY");
  }
  return true;
}

#if KMP_GROUP_AFFINITY
// If multiple Windows* OS processor groups exist, we can create a 2-level
// topology map with the groups at level 0 and the individual procs at level 1.
// This facilitates letting the threads float among all procs in a group,
// if granularity=group (the default when there are multiple groups).
static bool __kmp_affinity_create_proc_group_map(kmp_i18n_id_t *const msg_id) {
  *msg_id = kmp_i18n_null;
  int depth = 3;
  kmp_hw_t types[] = {KMP_HW_PROC_GROUP, KMP_HW_CORE, KMP_HW_THREAD};
  const static size_t BITS_PER_GROUP = CHAR_BIT * sizeof(DWORD_PTR);

  if (__kmp_affinity.flags.verbose) {
    KMP_INFORM(AffWindowsProcGroupMap, "KMP_AFFINITY");
  }

  // If we aren't affinity capable, then use flat topology
  if (!KMP_AFFINITY_CAPABLE()) {
    KMP_ASSERT(__kmp_affinity.type == affinity_none);
    nPackages = __kmp_num_proc_groups;
    __kmp_nThreadsPerCore = 1;
    __kmp_ncores = __kmp_xproc;
    nCoresPerPkg = nPackages / __kmp_ncores;
    return true;
  }

  // Construct the data structure to be returned.
  __kmp_topology = kmp_topology_t::allocate(__kmp_avail_proc, depth, types);
  int avail_ct = 0;
  int i;
  KMP_CPU_SET_ITERATE(i, __kmp_affin_fullMask) {
    // Skip this proc if it is not included in the machine model.
    if (!KMP_CPU_ISSET(i, __kmp_affin_fullMask)) {
      continue;
    }
    kmp_hw_thread_t &hw_thread = __kmp_topology->at(avail_ct++);
    hw_thread.clear();
    hw_thread.os_id = i;
    hw_thread.ids[0] = i / BITS_PER_GROUP;
    hw_thread.ids[1] = hw_thread.ids[2] = i % BITS_PER_GROUP;
  }
  return true;
}
#endif /* KMP_GROUP_AFFINITY */

#if KMP_ARCH_X86 || KMP_ARCH_X86_64

template <kmp_uint32 LSB, kmp_uint32 MSB>
static inline unsigned __kmp_extract_bits(kmp_uint32 v) {
  const kmp_uint32 SHIFT_LEFT = sizeof(kmp_uint32) * 8 - 1 - MSB;
  const kmp_uint32 SHIFT_RIGHT = LSB;
  kmp_uint32 retval = v;
  retval <<= SHIFT_LEFT;
  retval >>= (SHIFT_LEFT + SHIFT_RIGHT);
  return retval;
}

static int __kmp_cpuid_mask_width(int count) {
  int r = 0;

  while ((1 << r) < count)
    ++r;
  return r;
}

class apicThreadInfo {
public:
  unsigned osId; // param to __kmp_affinity_bind_thread
  unsigned apicId; // from cpuid after binding
  unsigned maxCoresPerPkg; //      ""
  unsigned maxThreadsPerPkg; //      ""
  unsigned pkgId; // inferred from above values
  unsigned coreId; //      ""
  unsigned threadId; //      ""
};

static int __kmp_affinity_cmp_apicThreadInfo_phys_id(const void *a,
                                                     const void *b) {
  const apicThreadInfo *aa = (const apicThreadInfo *)a;
  const apicThreadInfo *bb = (const apicThreadInfo *)b;
  if (aa->pkgId < bb->pkgId)
    return -1;
  if (aa->pkgId > bb->pkgId)
    return 1;
  if (aa->coreId < bb->coreId)
    return -1;
  if (aa->coreId > bb->coreId)
    return 1;
  if (aa->threadId < bb->threadId)
    return -1;
  if (aa->threadId > bb->threadId)
    return 1;
  return 0;
}

class kmp_cache_info_t {
public:
  struct info_t {
    unsigned level, mask;
  };
  kmp_cache_info_t() : depth(0) { get_leaf4_levels(); }
  size_t get_depth() const { return depth; }
  info_t &operator[](size_t index) { return table[index]; }
  const info_t &operator[](size_t index) const { return table[index]; }

  static kmp_hw_t get_topology_type(unsigned level) {
    KMP_DEBUG_ASSERT(level >= 1 && level <= MAX_CACHE_LEVEL);
    switch (level) {
    case 1:
      return KMP_HW_L1;
    case 2:
      return KMP_HW_L2;
    case 3:
      return KMP_HW_L3;
    }
    return KMP_HW_UNKNOWN;
  }

private:
  static const int MAX_CACHE_LEVEL = 3;

  size_t depth;
  info_t table[MAX_CACHE_LEVEL];

  void get_leaf4_levels() {
    unsigned level = 0;
    while (depth < MAX_CACHE_LEVEL) {
      unsigned cache_type, max_threads_sharing;
      unsigned cache_level, cache_mask_width;
      kmp_cpuid buf2;
      __kmp_x86_cpuid(4, level, &buf2);
      cache_type = __kmp_extract_bits<0, 4>(buf2.eax);
      if (!cache_type)
        break;
      // Skip instruction caches
      if (cache_type == 2) {
        level++;
        continue;
      }
      max_threads_sharing = __kmp_extract_bits<14, 25>(buf2.eax) + 1;
      cache_mask_width = __kmp_cpuid_mask_width(max_threads_sharing);
      cache_level = __kmp_extract_bits<5, 7>(buf2.eax);
      table[depth].level = cache_level;
      table[depth].mask = ((-1) << cache_mask_width);
      depth++;
      level++;
    }
  }
};

// On IA-32 architecture and Intel(R) 64 architecture, we attempt to use
// an algorithm which cycles through the available os threads, setting
// the current thread's affinity mask to that thread, and then retrieves
// the Apic Id for each thread context using the cpuid instruction.
static bool __kmp_affinity_create_apicid_map(kmp_i18n_id_t *const msg_id) {
  kmp_cpuid buf;
  *msg_id = kmp_i18n_null;

  if (__kmp_affinity.flags.verbose) {
    KMP_INFORM(AffInfoStr, "KMP_AFFINITY", KMP_I18N_STR(DecodingLegacyAPIC));
  }

  // Check if cpuid leaf 4 is supported.
  __kmp_x86_cpuid(0, 0, &buf);
  if (buf.eax < 4) {
    *msg_id = kmp_i18n_str_NoLeaf4Support;
    return false;
  }

  // The algorithm used starts by setting the affinity to each available thread
  // and retrieving info from the cpuid instruction, so if we are not capable of
  // calling __kmp_get_system_affinity() and _kmp_get_system_affinity(), then we
  // need to do something else - use the defaults that we calculated from
  // issuing cpuid without binding to each proc.
  if (!KMP_AFFINITY_CAPABLE()) {
    // Hack to try and infer the machine topology using only the data
    // available from cpuid on the current thread, and __kmp_xproc.
    KMP_ASSERT(__kmp_affinity.type == affinity_none);

    // Get an upper bound on the number of threads per package using cpuid(1).
    // On some OS/chps combinations where HT is supported by the chip but is
    // disabled, this value will be 2 on a single core chip. Usually, it will be
    // 2 if HT is enabled and 1 if HT is disabled.
    __kmp_x86_cpuid(1, 0, &buf);
    int maxThreadsPerPkg = (buf.ebx >> 16) & 0xff;
    if (maxThreadsPerPkg == 0) {
      maxThreadsPerPkg = 1;
    }

    // The num cores per pkg comes from cpuid(4). 1 must be added to the encoded
    // value.
    //
    // The author of cpu_count.cpp treated this only an upper bound on the
    // number of cores, but I haven't seen any cases where it was greater than
    // the actual number of cores, so we will treat it as exact in this block of
    // code.
    //
    // First, we need to check if cpuid(4) is supported on this chip. To see if
    // cpuid(n) is supported, issue cpuid(0) and check if eax has the value n or
    // greater.
    __kmp_x86_cpuid(0, 0, &buf);
    if (buf.eax >= 4) {
      __kmp_x86_cpuid(4, 0, &buf);
      nCoresPerPkg = ((buf.eax >> 26) & 0x3f) + 1;
    } else {
      nCoresPerPkg = 1;
    }

    // There is no way to reliably tell if HT is enabled without issuing the
    // cpuid instruction from every thread, can correlating the cpuid info, so
    // if the machine is not affinity capable, we assume that HT is off. We have
    // seen quite a few machines where maxThreadsPerPkg is 2, yet the machine
    // does not support HT.
    //
    // - Older OSes are usually found on machines with older chips, which do not
    //   support HT.
    // - The performance penalty for mistakenly identifying a machine as HT when
    //   it isn't (which results in blocktime being incorrectly set to 0) is
    //   greater than the penalty when for mistakenly identifying a machine as
    //   being 1 thread/core when it is really HT enabled (which results in
    //   blocktime being incorrectly set to a positive value).
    __kmp_ncores = __kmp_xproc;
    nPackages = (__kmp_xproc + nCoresPerPkg - 1) / nCoresPerPkg;
    __kmp_nThreadsPerCore = 1;
    return true;
  }

  // From here on, we can assume that it is safe to call
  // __kmp_get_system_affinity() and __kmp_set_system_affinity(), even if
  // __kmp_affinity.type = affinity_none.

  // Save the affinity mask for the current thread.
  kmp_affinity_raii_t previous_affinity;

  // Run through each of the available contexts, binding the current thread
  // to it, and obtaining the pertinent information using the cpuid instr.
  //
  // The relevant information is:
  // - Apic Id: Bits 24:31 of ebx after issuing cpuid(1) - each thread context
  //     has a uniqie Apic Id, which is of the form pkg# : core# : thread#.
  // - Max Threads Per Pkg: Bits 16:23 of ebx after issuing cpuid(1). The value
  //     of this field determines the width of the core# + thread# fields in the
  //     Apic Id. It is also an upper bound on the number of threads per
  //     package, but it has been verified that situations happen were it is not
  //     exact. In particular, on certain OS/chip combinations where Intel(R)
  //     Hyper-Threading Technology is supported by the chip but has been
  //     disabled, the value of this field will be 2 (for a single core chip).
  //     On other OS/chip combinations supporting Intel(R) Hyper-Threading
  //     Technology, the value of this field will be 1 when Intel(R)
  //     Hyper-Threading Technology is disabled and 2 when it is enabled.
  // - Max Cores Per Pkg:  Bits 26:31 of eax after issuing cpuid(4). The value
  //     of this field (+1) determines the width of the core# field in the Apic
  //     Id. The comments in "cpucount.cpp" say that this value is an upper
  //     bound, but the IA-32 architecture manual says that it is exactly the
  //     number of cores per package, and I haven't seen any case where it
  //     wasn't.
  //
  // From this information, deduce the package Id, core Id, and thread Id,
  // and set the corresponding fields in the apicThreadInfo struct.
  unsigned i;
  apicThreadInfo *threadInfo = (apicThreadInfo *)__kmp_allocate(
      __kmp_avail_proc * sizeof(apicThreadInfo));
  unsigned nApics = 0;
  KMP_CPU_SET_ITERATE(i, __kmp_affin_fullMask) {
    // Skip this proc if it is not included in the machine model.
    if (!KMP_CPU_ISSET(i, __kmp_affin_fullMask)) {
      continue;
    }
    KMP_DEBUG_ASSERT((int)nApics < __kmp_avail_proc);

    __kmp_affinity_dispatch->bind_thread(i);
    threadInfo[nApics].osId = i;

    // The apic id and max threads per pkg come from cpuid(1).
    __kmp_x86_cpuid(1, 0, &buf);
    if (((buf.edx >> 9) & 1) == 0) {
      __kmp_free(threadInfo);
      *msg_id = kmp_i18n_str_ApicNotPresent;
      return false;
    }
    threadInfo[nApics].apicId = (buf.ebx >> 24) & 0xff;
    threadInfo[nApics].maxThreadsPerPkg = (buf.ebx >> 16) & 0xff;
    if (threadInfo[nApics].maxThreadsPerPkg == 0) {
      threadInfo[nApics].maxThreadsPerPkg = 1;
    }

    // Max cores per pkg comes from cpuid(4). 1 must be added to the encoded
    // value.
    //
    // First, we need to check if cpuid(4) is supported on this chip. To see if
    // cpuid(n) is supported, issue cpuid(0) and check if eax has the value n
    // or greater.
    __kmp_x86_cpuid(0, 0, &buf);
    if (buf.eax >= 4) {
      __kmp_x86_cpuid(4, 0, &buf);
      threadInfo[nApics].maxCoresPerPkg = ((buf.eax >> 26) & 0x3f) + 1;
    } else {
      threadInfo[nApics].maxCoresPerPkg = 1;
    }

    // Infer the pkgId / coreId / threadId using only the info obtained locally.
    int widthCT = __kmp_cpuid_mask_width(threadInfo[nApics].maxThreadsPerPkg);
    threadInfo[nApics].pkgId = threadInfo[nApics].apicId >> widthCT;

    int widthC = __kmp_cpuid_mask_width(threadInfo[nApics].maxCoresPerPkg);
    int widthT = widthCT - widthC;
    if (widthT < 0) {
      // I've never seen this one happen, but I suppose it could, if the cpuid
      // instruction on a chip was really screwed up. Make sure to restore the
      // affinity mask before the tail call.
      __kmp_free(threadInfo);
      *msg_id = kmp_i18n_str_InvalidCpuidInfo;
      return false;
    }

    int maskC = (1 << widthC) - 1;
    threadInfo[nApics].coreId = (threadInfo[nApics].apicId >> widthT) & maskC;

    int maskT = (1 << widthT) - 1;
    threadInfo[nApics].threadId = threadInfo[nApics].apicId & maskT;

    nApics++;
  }

  // We've collected all the info we need.
  // Restore the old affinity mask for this thread.
  previous_affinity.restore();

  // Sort the threadInfo table by physical Id.
  qsort(threadInfo, nApics, sizeof(*threadInfo),
        __kmp_affinity_cmp_apicThreadInfo_phys_id);

  // The table is now sorted by pkgId / coreId / threadId, but we really don't
  // know the radix of any of the fields. pkgId's may be sparsely assigned among
  // the chips on a system. Although coreId's are usually assigned
  // [0 .. coresPerPkg-1] and threadId's are usually assigned
  // [0..threadsPerCore-1], we don't want to make any such assumptions.
  //
  // For that matter, we don't know what coresPerPkg and threadsPerCore (or the
  // total # packages) are at this point - we want to determine that now. We
  // only have an upper bound on the first two figures.
  //
  // We also perform a consistency check at this point: the values returned by
  // the cpuid instruction for any thread bound to a given package had better
  // return the same info for maxThreadsPerPkg and maxCoresPerPkg.
  nPackages = 1;
  nCoresPerPkg = 1;
  __kmp_nThreadsPerCore = 1;
  unsigned nCores = 1;

  unsigned pkgCt = 1; // to determine radii
  unsigned lastPkgId = threadInfo[0].pkgId;
  unsigned coreCt = 1;
  unsigned lastCoreId = threadInfo[0].coreId;
  unsigned threadCt = 1;
  unsigned lastThreadId = threadInfo[0].threadId;

  // intra-pkg consist checks
  unsigned prevMaxCoresPerPkg = threadInfo[0].maxCoresPerPkg;
  unsigned prevMaxThreadsPerPkg = threadInfo[0].maxThreadsPerPkg;

  for (i = 1; i < nApics; i++) {
    if (threadInfo[i].pkgId != lastPkgId) {
      nCores++;
      pkgCt++;
      lastPkgId = threadInfo[i].pkgId;
      if ((int)coreCt > nCoresPerPkg)
        nCoresPerPkg = coreCt;
      coreCt = 1;
      lastCoreId = threadInfo[i].coreId;
      if ((int)threadCt > __kmp_nThreadsPerCore)
        __kmp_nThreadsPerCore = threadCt;
      threadCt = 1;
      lastThreadId = threadInfo[i].threadId;

      // This is a different package, so go on to the next iteration without
      // doing any consistency checks. Reset the consistency check vars, though.
      prevMaxCoresPerPkg = threadInfo[i].maxCoresPerPkg;
      prevMaxThreadsPerPkg = threadInfo[i].maxThreadsPerPkg;
      continue;
    }

    if (threadInfo[i].coreId != lastCoreId) {
      nCores++;
      coreCt++;
      lastCoreId = threadInfo[i].coreId;
      if ((int)threadCt > __kmp_nThreadsPerCore)
        __kmp_nThreadsPerCore = threadCt;
      threadCt = 1;
      lastThreadId = threadInfo[i].threadId;
    } else if (threadInfo[i].threadId != lastThreadId) {
      threadCt++;
      lastThreadId = threadInfo[i].threadId;
    } else {
      __kmp_free(threadInfo);
      *msg_id = kmp_i18n_str_LegacyApicIDsNotUnique;
      return false;
    }

    // Check to make certain that the maxCoresPerPkg and maxThreadsPerPkg
    // fields agree between all the threads bounds to a given package.
    if ((prevMaxCoresPerPkg != threadInfo[i].maxCoresPerPkg) ||
        (prevMaxThreadsPerPkg != threadInfo[i].maxThreadsPerPkg)) {
      __kmp_free(threadInfo);
      *msg_id = kmp_i18n_str_InconsistentCpuidInfo;
      return false;
    }
  }
  // When affinity is off, this routine will still be called to set
  // __kmp_ncores, as well as __kmp_nThreadsPerCore, nCoresPerPkg, & nPackages.
  // Make sure all these vars are set correctly
  nPackages = pkgCt;
  if ((int)coreCt > nCoresPerPkg)
    nCoresPerPkg = coreCt;
  if ((int)threadCt > __kmp_nThreadsPerCore)
    __kmp_nThreadsPerCore = threadCt;
  __kmp_ncores = nCores;
  KMP_DEBUG_ASSERT(nApics == (unsigned)__kmp_avail_proc);

  // Now that we've determined the number of packages, the number of cores per
  // package, and the number of threads per core, we can construct the data
  // structure that is to be returned.
  int idx = 0;
  int pkgLevel = 0;
  int coreLevel = 1;
  int threadLevel = 2;
  //(__kmp_nThreadsPerCore <= 1) ? -1 : ((coreLevel >= 0) ? 2 : 1);
  int depth = (pkgLevel >= 0) + (coreLevel >= 0) + (threadLevel >= 0);
  kmp_hw_t types[3];
  if (pkgLevel >= 0)
    types[idx++] = KMP_HW_SOCKET;
  if (coreLevel >= 0)
    types[idx++] = KMP_HW_CORE;
  if (threadLevel >= 0)
    types[idx++] = KMP_HW_THREAD;

  KMP_ASSERT(depth > 0);
  __kmp_topology = kmp_topology_t::allocate(nApics, depth, types);

  for (i = 0; i < nApics; ++i) {
    idx = 0;
    unsigned os = threadInfo[i].osId;
    kmp_hw_thread_t &hw_thread = __kmp_topology->at(i);
    hw_thread.clear();

    if (pkgLevel >= 0) {
      hw_thread.ids[idx++] = threadInfo[i].pkgId;
    }
    if (coreLevel >= 0) {
      hw_thread.ids[idx++] = threadInfo[i].coreId;
    }
    if (threadLevel >= 0) {
      hw_thread.ids[idx++] = threadInfo[i].threadId;
    }
    hw_thread.os_id = os;
  }

  __kmp_free(threadInfo);
  __kmp_topology->sort_ids();
  if (!__kmp_topology->check_ids()) {
    kmp_topology_t::deallocate(__kmp_topology);
    __kmp_topology = nullptr;
    *msg_id = kmp_i18n_str_LegacyApicIDsNotUnique;
    return false;
  }
  return true;
}

// Hybrid cpu detection using CPUID.1A
// Thread should be pinned to processor already
static void __kmp_get_hybrid_info(kmp_hw_core_type_t *type, int *efficiency,
                                  unsigned *native_model_id) {
  kmp_cpuid buf;
  __kmp_x86_cpuid(0x1a, 0, &buf);
  *type = (kmp_hw_core_type_t)__kmp_extract_bits<24, 31>(buf.eax);
  switch (*type) {
  case KMP_HW_CORE_TYPE_ATOM:
    *efficiency = 0;
    break;
  case KMP_HW_CORE_TYPE_CORE:
    *efficiency = 1;
    break;
  default:
    *efficiency = 0;
  }
  *native_model_id = __kmp_extract_bits<0, 23>(buf.eax);
}

// Intel(R) microarchitecture code name Nehalem, Dunnington and later
// architectures support a newer interface for specifying the x2APIC Ids,
// based on CPUID.B or CPUID.1F
/*
 * CPUID.B or 1F, Input ECX (sub leaf # aka level number)
    Bits            Bits            Bits           Bits
    31-16           15-8            7-4            4-0
---+-----------+--------------+-------------+-----------------+
EAX| reserved  |   reserved   |   reserved  |  Bits to Shift  |
---+-----------|--------------+-------------+-----------------|
EBX| reserved  | Num logical processors at level (16 bits)    |
---+-----------|--------------+-------------------------------|
ECX| reserved  |   Level Type |      Level Number (8 bits)    |
---+-----------+--------------+-------------------------------|
EDX|                    X2APIC ID (32 bits)                   |
---+----------------------------------------------------------+
*/

enum {
  INTEL_LEVEL_TYPE_INVALID = 0, // Package level
  INTEL_LEVEL_TYPE_SMT = 1,
  INTEL_LEVEL_TYPE_CORE = 2,
  INTEL_LEVEL_TYPE_MODULE = 3,
  INTEL_LEVEL_TYPE_TILE = 4,
  INTEL_LEVEL_TYPE_DIE = 5,
  INTEL_LEVEL_TYPE_LAST = 6,
};

struct cpuid_level_info_t {
  unsigned level_type, mask, mask_width, nitems, cache_mask;
};

static kmp_hw_t __kmp_intel_type_2_topology_type(int intel_type) {
  switch (intel_type) {
  case INTEL_LEVEL_TYPE_INVALID:
    return KMP_HW_SOCKET;
  case INTEL_LEVEL_TYPE_SMT:
    return KMP_HW_THREAD;
  case INTEL_LEVEL_TYPE_CORE:
    return KMP_HW_CORE;
  case INTEL_LEVEL_TYPE_TILE:
    return KMP_HW_TILE;
  case INTEL_LEVEL_TYPE_MODULE:
    return KMP_HW_MODULE;
  case INTEL_LEVEL_TYPE_DIE:
    return KMP_HW_DIE;
  }
  return KMP_HW_UNKNOWN;
}

// This function takes the topology leaf, a levels array to store the levels
// detected and a bitmap of the known levels.
// Returns the number of levels in the topology
static unsigned
__kmp_x2apicid_get_levels(int leaf,
                          cpuid_level_info_t levels[INTEL_LEVEL_TYPE_LAST],
                          kmp_uint64 known_levels) {
  unsigned level, levels_index;
  unsigned level_type, mask_width, nitems;
  kmp_cpuid buf;

  // New algorithm has known topology layers act as highest unknown topology
  // layers when unknown topology layers exist.
  // e.g., Suppose layers were SMT <X> CORE <Y> <Z> PACKAGE, where <X> <Y> <Z>
  // are unknown topology layers, Then SMT will take the characteristics of
  // (SMT x <X>) and CORE will take the characteristics of (CORE x <Y> x <Z>).
  // This eliminates unknown portions of the topology while still keeping the
  // correct structure.
  level = levels_index = 0;
  do {
    __kmp_x86_cpuid(leaf, level, &buf);
    level_type = __kmp_extract_bits<8, 15>(buf.ecx);
    mask_width = __kmp_extract_bits<0, 4>(buf.eax);
    nitems = __kmp_extract_bits<0, 15>(buf.ebx);
    if (level_type != INTEL_LEVEL_TYPE_INVALID && nitems == 0)
      return 0;

    if (known_levels & (1ull << level_type)) {
      // Add a new level to the topology
      KMP_ASSERT(levels_index < INTEL_LEVEL_TYPE_LAST);
      levels[levels_index].level_type = level_type;
      levels[levels_index].mask_width = mask_width;
      levels[levels_index].nitems = nitems;
      levels_index++;
    } else {
      // If it is an unknown level, then logically move the previous layer up
      if (levels_index > 0) {
        levels[levels_index - 1].mask_width = mask_width;
        levels[levels_index - 1].nitems = nitems;
      }
    }
    level++;
  } while (level_type != INTEL_LEVEL_TYPE_INVALID);

  // Ensure the INTEL_LEVEL_TYPE_INVALID (Socket) layer isn't first
  if (levels_index == 0 || levels[0].level_type == INTEL_LEVEL_TYPE_INVALID)
    return 0;

  // Set the masks to & with apicid
  for (unsigned i = 0; i < levels_index; ++i) {
    if (levels[i].level_type != INTEL_LEVEL_TYPE_INVALID) {
      levels[i].mask = ~((-1) << levels[i].mask_width);
      levels[i].cache_mask = (-1) << levels[i].mask_width;
      for (unsigned j = 0; j < i; ++j)
        levels[i].mask ^= levels[j].mask;
    } else {
      KMP_DEBUG_ASSERT(i > 0);
      levels[i].mask = (-1) << levels[i - 1].mask_width;
      levels[i].cache_mask = 0;
    }
  }
  return levels_index;
}

static bool __kmp_affinity_create_x2apicid_map(kmp_i18n_id_t *const msg_id) {

  cpuid_level_info_t levels[INTEL_LEVEL_TYPE_LAST];
  kmp_hw_t types[INTEL_LEVEL_TYPE_LAST];
  unsigned levels_index;
  kmp_cpuid buf;
  kmp_uint64 known_levels;
  int topology_leaf, highest_leaf, apic_id;
  int num_leaves;
  static int leaves[] = {0, 0};

  kmp_i18n_id_t leaf_message_id;

  KMP_BUILD_ASSERT(sizeof(known_levels) * CHAR_BIT > KMP_HW_LAST);

  *msg_id = kmp_i18n_null;
  if (__kmp_affinity.flags.verbose) {
    KMP_INFORM(AffInfoStr, "KMP_AFFINITY", KMP_I18N_STR(Decodingx2APIC));
  }

  // Figure out the known topology levels
  known_levels = 0ull;
  for (int i = 0; i < INTEL_LEVEL_TYPE_LAST; ++i) {
    if (__kmp_intel_type_2_topology_type(i) != KMP_HW_UNKNOWN) {
      known_levels |= (1ull << i);
    }
  }

  // Get the highest cpuid leaf supported
  __kmp_x86_cpuid(0, 0, &buf);
  highest_leaf = buf.eax;

  // If a specific topology method was requested, only allow that specific leaf
  // otherwise, try both leaves 31 and 11 in that order
  num_leaves = 0;
  if (__kmp_affinity_top_method == affinity_top_method_x2apicid) {
    num_leaves = 1;
    leaves[0] = 11;
    leaf_message_id = kmp_i18n_str_NoLeaf11Support;
  } else if (__kmp_affinity_top_method == affinity_top_method_x2apicid_1f) {
    num_leaves = 1;
    leaves[0] = 31;
    leaf_message_id = kmp_i18n_str_NoLeaf31Support;
  } else {
    num_leaves = 2;
    leaves[0] = 31;
    leaves[1] = 11;
    leaf_message_id = kmp_i18n_str_NoLeaf11Support;
  }

  // Check to see if cpuid leaf 31 or 11 is supported.
  __kmp_nThreadsPerCore = nCoresPerPkg = nPackages = 1;
  topology_leaf = -1;
  for (int i = 0; i < num_leaves; ++i) {
    int leaf = leaves[i];
    if (highest_leaf < leaf)
      continue;
    __kmp_x86_cpuid(leaf, 0, &buf);
    if (buf.ebx == 0)
      continue;
    topology_leaf = leaf;
    levels_index = __kmp_x2apicid_get_levels(leaf, levels, known_levels);
    if (levels_index == 0)
      continue;
    break;
  }
  if (topology_leaf == -1 || levels_index == 0) {
    *msg_id = leaf_message_id;
    return false;
  }
  KMP_ASSERT(levels_index <= INTEL_LEVEL_TYPE_LAST);

  // The algorithm used starts by setting the affinity to each available thread
  // and retrieving info from the cpuid instruction, so if we are not capable of
  // calling __kmp_get_system_affinity() and __kmp_get_system_affinity(), then
  // we need to do something else - use the defaults that we calculated from
  // issuing cpuid without binding to each proc.
  if (!KMP_AFFINITY_CAPABLE()) {
    // Hack to try and infer the machine topology using only the data
    // available from cpuid on the current thread, and __kmp_xproc.
    KMP_ASSERT(__kmp_affinity.type == affinity_none);
    for (unsigned i = 0; i < levels_index; ++i) {
      if (levels[i].level_type == INTEL_LEVEL_TYPE_SMT) {
        __kmp_nThreadsPerCore = levels[i].nitems;
      } else if (levels[i].level_type == INTEL_LEVEL_TYPE_CORE) {
        nCoresPerPkg = levels[i].nitems;
      }
    }
    __kmp_ncores = __kmp_xproc / __kmp_nThreadsPerCore;
    nPackages = (__kmp_xproc + nCoresPerPkg - 1) / nCoresPerPkg;
    return true;
  }

  // Allocate the data structure to be returned.
  int depth = levels_index;
  for (int i = depth - 1, j = 0; i >= 0; --i, ++j)
    types[j] = __kmp_intel_type_2_topology_type(levels[i].level_type);
  __kmp_topology =
      kmp_topology_t::allocate(__kmp_avail_proc, levels_index, types);

  // Insert equivalent cache types if they exist
  kmp_cache_info_t cache_info;
  for (size_t i = 0; i < cache_info.get_depth(); ++i) {
    const kmp_cache_info_t::info_t &info = cache_info[i];
    unsigned cache_mask = info.mask;
    unsigned cache_level = info.level;
    for (unsigned j = 0; j < levels_index; ++j) {
      unsigned hw_cache_mask = levels[j].cache_mask;
      kmp_hw_t cache_type = kmp_cache_info_t::get_topology_type(cache_level);
      if (hw_cache_mask == cache_mask && j < levels_index - 1) {
        kmp_hw_t type =
            __kmp_intel_type_2_topology_type(levels[j + 1].level_type);
        __kmp_topology->set_equivalent_type(cache_type, type);
      }
    }
  }

  // From here on, we can assume that it is safe to call
  // __kmp_get_system_affinity() and __kmp_set_system_affinity(), even if
  // __kmp_affinity.type = affinity_none.

  // Save the affinity mask for the current thread.
  kmp_affinity_raii_t previous_affinity;

  // Run through each of the available contexts, binding the current thread
  // to it, and obtaining the pertinent information using the cpuid instr.
  unsigned int proc;
  int hw_thread_index = 0;
  KMP_CPU_SET_ITERATE(proc, __kmp_affin_fullMask) {
    cpuid_level_info_t my_levels[INTEL_LEVEL_TYPE_LAST];
    unsigned my_levels_index;

    // Skip this proc if it is not included in the machine model.
    if (!KMP_CPU_ISSET(proc, __kmp_affin_fullMask)) {
      continue;
    }
    KMP_DEBUG_ASSERT(hw_thread_index < __kmp_avail_proc);

    __kmp_affinity_dispatch->bind_thread(proc);

    // New algorithm
    __kmp_x86_cpuid(topology_leaf, 0, &buf);
    apic_id = buf.edx;
    kmp_hw_thread_t &hw_thread = __kmp_topology->at(hw_thread_index);
    my_levels_index =
        __kmp_x2apicid_get_levels(topology_leaf, my_levels, known_levels);
    if (my_levels_index == 0 || my_levels_index != levels_index) {
      *msg_id = kmp_i18n_str_InvalidCpuidInfo;
      return false;
    }
    hw_thread.clear();
    hw_thread.os_id = proc;
    // Put in topology information
    for (unsigned j = 0, idx = depth - 1; j < my_levels_index; ++j, --idx) {
      hw_thread.ids[idx] = apic_id & my_levels[j].mask;
      if (j > 0) {
        hw_thread.ids[idx] >>= my_levels[j - 1].mask_width;
      }
    }
    // Hybrid information
    if (__kmp_is_hybrid_cpu() && highest_leaf >= 0x1a) {
      kmp_hw_core_type_t type;
      unsigned native_model_id;
      int efficiency;
      __kmp_get_hybrid_info(&type, &efficiency, &native_model_id);
      hw_thread.attrs.set_core_type(type);
      hw_thread.attrs.set_core_eff(efficiency);
    }
    hw_thread_index++;
  }
  KMP_ASSERT(hw_thread_index > 0);
  __kmp_topology->sort_ids();
  if (!__kmp_topology->check_ids()) {
    kmp_topology_t::deallocate(__kmp_topology);
    __kmp_topology = nullptr;
    *msg_id = kmp_i18n_str_x2ApicIDsNotUnique;
    return false;
  }
  return true;
}
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

#define osIdIndex 0
#define threadIdIndex 1
#define coreIdIndex 2
#define pkgIdIndex 3
#define nodeIdIndex 4

typedef unsigned *ProcCpuInfo;
static unsigned maxIndex = pkgIdIndex;

static int __kmp_affinity_cmp_ProcCpuInfo_phys_id(const void *a,
                                                  const void *b) {
  unsigned i;
  const unsigned *aa = *(unsigned *const *)a;
  const unsigned *bb = *(unsigned *const *)b;
  for (i = maxIndex;; i--) {
    if (aa[i] < bb[i])
      return -1;
    if (aa[i] > bb[i])
      return 1;
    if (i == osIdIndex)
      break;
  }
  return 0;
}

#if KMP_USE_HIER_SCHED
// Set the array sizes for the hierarchy layers
static void __kmp_dispatch_set_hierarchy_values() {
  // Set the maximum number of L1's to number of cores
  // Set the maximum number of L2's to either number of cores / 2 for
  // Intel(R) Xeon Phi(TM) coprocessor formally codenamed Knights Landing
  // Or the number of cores for Intel(R) Xeon(R) processors
  // Set the maximum number of NUMA nodes and L3's to number of packages
  __kmp_hier_max_units[kmp_hier_layer_e::LAYER_THREAD + 1] =
      nPackages * nCoresPerPkg * __kmp_nThreadsPerCore;
  __kmp_hier_max_units[kmp_hier_layer_e::LAYER_L1 + 1] = __kmp_ncores;
#if KMP_ARCH_X86_64 && (KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_WINDOWS) &&   \
    KMP_MIC_SUPPORTED
  if (__kmp_mic_type >= mic3)
    __kmp_hier_max_units[kmp_hier_layer_e::LAYER_L2 + 1] = __kmp_ncores / 2;
  else
#endif // KMP_ARCH_X86_64 && (KMP_OS_LINUX || KMP_OS_WINDOWS)
    __kmp_hier_max_units[kmp_hier_layer_e::LAYER_L2 + 1] = __kmp_ncores;
  __kmp_hier_max_units[kmp_hier_layer_e::LAYER_L3 + 1] = nPackages;
  __kmp_hier_max_units[kmp_hier_layer_e::LAYER_NUMA + 1] = nPackages;
  __kmp_hier_max_units[kmp_hier_layer_e::LAYER_LOOP + 1] = 1;
  // Set the number of threads per unit
  // Number of hardware threads per L1/L2/L3/NUMA/LOOP
  __kmp_hier_threads_per[kmp_hier_layer_e::LAYER_THREAD + 1] = 1;
  __kmp_hier_threads_per[kmp_hier_layer_e::LAYER_L1 + 1] =
      __kmp_nThreadsPerCore;
#if KMP_ARCH_X86_64 && (KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_WINDOWS) &&   \
    KMP_MIC_SUPPORTED
  if (__kmp_mic_type >= mic3)
    __kmp_hier_threads_per[kmp_hier_layer_e::LAYER_L2 + 1] =
        2 * __kmp_nThreadsPerCore;
  else
#endif // KMP_ARCH_X86_64 && (KMP_OS_LINUX || KMP_OS_WINDOWS)
    __kmp_hier_threads_per[kmp_hier_layer_e::LAYER_L2 + 1] =
        __kmp_nThreadsPerCore;
  __kmp_hier_threads_per[kmp_hier_layer_e::LAYER_L3 + 1] =
      nCoresPerPkg * __kmp_nThreadsPerCore;
  __kmp_hier_threads_per[kmp_hier_layer_e::LAYER_NUMA + 1] =
      nCoresPerPkg * __kmp_nThreadsPerCore;
  __kmp_hier_threads_per[kmp_hier_layer_e::LAYER_LOOP + 1] =
      nPackages * nCoresPerPkg * __kmp_nThreadsPerCore;
}

// Return the index into the hierarchy for this tid and layer type (L1, L2, etc)
// i.e., this thread's L1 or this thread's L2, etc.
int __kmp_dispatch_get_index(int tid, kmp_hier_layer_e type) {
  int index = type + 1;
  int num_hw_threads = __kmp_hier_max_units[kmp_hier_layer_e::LAYER_THREAD + 1];
  KMP_DEBUG_ASSERT(type != kmp_hier_layer_e::LAYER_LAST);
  if (type == kmp_hier_layer_e::LAYER_THREAD)
    return tid;
  else if (type == kmp_hier_layer_e::LAYER_LOOP)
    return 0;
  KMP_DEBUG_ASSERT(__kmp_hier_max_units[index] != 0);
  if (tid >= num_hw_threads)
    tid = tid % num_hw_threads;
  return (tid / __kmp_hier_threads_per[index]) % __kmp_hier_max_units[index];
}

// Return the number of t1's per t2
int __kmp_dispatch_get_t1_per_t2(kmp_hier_layer_e t1, kmp_hier_layer_e t2) {
  int i1 = t1 + 1;
  int i2 = t2 + 1;
  KMP_DEBUG_ASSERT(i1 <= i2);
  KMP_DEBUG_ASSERT(t1 != kmp_hier_layer_e::LAYER_LAST);
  KMP_DEBUG_ASSERT(t2 != kmp_hier_layer_e::LAYER_LAST);
  KMP_DEBUG_ASSERT(__kmp_hier_threads_per[i1] != 0);
  // (nthreads/t2) / (nthreads/t1) = t1 / t2
  return __kmp_hier_threads_per[i2] / __kmp_hier_threads_per[i1];
}
#endif // KMP_USE_HIER_SCHED

static inline const char *__kmp_cpuinfo_get_filename() {
  const char *filename;
  if (__kmp_cpuinfo_file != nullptr)
    filename = __kmp_cpuinfo_file;
  else
    filename = "/proc/cpuinfo";
  return filename;
}

static inline const char *__kmp_cpuinfo_get_envvar() {
  const char *envvar = nullptr;
  if (__kmp_cpuinfo_file != nullptr)
    envvar = "KMP_CPUINFO_FILE";
  return envvar;
}

// Parse /proc/cpuinfo (or an alternate file in the same format) to obtain the
// affinity map. On AIX, the map is obtained through system SRAD (Scheduler
// Resource Allocation Domain).
static bool __kmp_affinity_create_cpuinfo_map(int *line,
                                              kmp_i18n_id_t *const msg_id) {
  *msg_id = kmp_i18n_null;

#if KMP_OS_AIX
  unsigned num_records = __kmp_xproc;
#else
  const char *filename = __kmp_cpuinfo_get_filename();
  const char *envvar = __kmp_cpuinfo_get_envvar();

  if (__kmp_affinity.flags.verbose) {
    KMP_INFORM(AffParseFilename, "KMP_AFFINITY", filename);
  }

  kmp_safe_raii_file_t f(filename, "r", envvar);

  // Scan of the file, and count the number of "processor" (osId) fields,
  // and find the highest value of <n> for a node_<n> field.
  char buf[256];
  unsigned num_records = 0;
  while (!feof(f)) {
    buf[sizeof(buf) - 1] = 1;
    if (!fgets(buf, sizeof(buf), f)) {
      // Read errors presumably because of EOF
      break;
    }

    char s1[] = "processor";
    if (strncmp(buf, s1, sizeof(s1) - 1) == 0) {
      num_records++;
      continue;
    }

    // FIXME - this will match "node_<n> <garbage>"
    unsigned level;
    if (KMP_SSCANF(buf, "node_%u id", &level) == 1) {
      // validate the input fisrt:
      if (level > (unsigned)__kmp_xproc) { // level is too big
        level = __kmp_xproc;
      }
      if (nodeIdIndex + level >= maxIndex) {
        maxIndex = nodeIdIndex + level;
      }
      continue;
    }
  }

  // Check for empty file / no valid processor records, or too many. The number
  // of records can't exceed the number of valid bits in the affinity mask.
  if (num_records == 0) {
    *msg_id = kmp_i18n_str_NoProcRecords;
    return false;
  }
  if (num_records > (unsigned)__kmp_xproc) {
    *msg_id = kmp_i18n_str_TooManyProcRecords;
    return false;
  }

  // Set the file pointer back to the beginning, so that we can scan the file
  // again, this time performing a full parse of the data. Allocate a vector of
  // ProcCpuInfo object, where we will place the data. Adding an extra element
  // at the end allows us to remove a lot of extra checks for termination
  // conditions.
  if (fseek(f, 0, SEEK_SET) != 0) {
    *msg_id = kmp_i18n_str_CantRewindCpuinfo;
    return false;
  }
#endif // KMP_OS_AIX

  // Allocate the array of records to store the proc info in.  The dummy
  // element at the end makes the logic in filling them out easier to code.
  unsigned **threadInfo =
      (unsigned **)__kmp_allocate((num_records + 1) * sizeof(unsigned *));
  unsigned i;
  for (i = 0; i <= num_records; i++) {
    threadInfo[i] =
        (unsigned *)__kmp_allocate((maxIndex + 1) * sizeof(unsigned));
  }

#define CLEANUP_THREAD_INFO                                                    \
  for (i = 0; i <= num_records; i++) {                                         \
    __kmp_free(threadInfo[i]);                                                 \
  }                                                                            \
  __kmp_free(threadInfo);

  // A value of UINT_MAX means that we didn't find the field
  unsigned __index;

#define INIT_PROC_INFO(p)                                                      \
  for (__index = 0; __index <= maxIndex; __index++) {                          \
    (p)[__index] = UINT_MAX;                                                   \
  }

  for (i = 0; i <= num_records; i++) {
    INIT_PROC_INFO(threadInfo[i]);
  }

#if KMP_OS_AIX
  int smt_threads;
  lpar_info_format1_t cpuinfo;
  unsigned num_avail = __kmp_xproc;

  if (__kmp_affinity.flags.verbose)
    KMP_INFORM(AffParseFilename, "KMP_AFFINITY", "system info for topology");

  // Get the number of SMT threads per core.
  int retval =
      lpar_get_info(LPAR_INFO_FORMAT1, &cpuinfo, sizeof(lpar_info_format1_t));
  if (!retval)
    smt_threads = cpuinfo.smt_threads;
  else {
    CLEANUP_THREAD_INFO;
    *msg_id = kmp_i18n_str_UnknownTopology;
    return false;
  }

  // Allocate a resource set containing available system resourses.
  rsethandle_t sys_rset = rs_alloc(RS_SYSTEM);
  if (sys_rset == NULL) {
    CLEANUP_THREAD_INFO;
    *msg_id = kmp_i18n_str_UnknownTopology;
    return false;
  }
  // Allocate a resource set for the SRAD info.
  rsethandle_t srad = rs_alloc(RS_EMPTY);
  if (srad == NULL) {
    rs_free(sys_rset);
    CLEANUP_THREAD_INFO;
    *msg_id = kmp_i18n_str_UnknownTopology;
    return false;
  }

  // Get the SRAD system detail level.
  int sradsdl = rs_getinfo(NULL, R_SRADSDL, 0);
  if (sradsdl < 0) {
    rs_free(sys_rset);
    rs_free(srad);
    CLEANUP_THREAD_INFO;
    *msg_id = kmp_i18n_str_UnknownTopology;
    return false;
  }
  // Get the number of RADs at that SRAD SDL.
  int num_rads = rs_numrads(sys_rset, sradsdl, 0);
  if (num_rads < 0) {
    rs_free(sys_rset);
    rs_free(srad);
    CLEANUP_THREAD_INFO;
    *msg_id = kmp_i18n_str_UnknownTopology;
    return false;
  }

  // Get the maximum number of procs that may be contained in a resource set.
  int max_procs = rs_getinfo(NULL, R_MAXPROCS, 0);
  if (max_procs < 0) {
    rs_free(sys_rset);
    rs_free(srad);
    CLEANUP_THREAD_INFO;
    *msg_id = kmp_i18n_str_UnknownTopology;
    return false;
  }

  int cur_rad = 0;
  int num_set = 0;
  for (int srad_idx = 0; cur_rad < num_rads && srad_idx < VMI_MAXRADS;
       ++srad_idx) {
    // Check if the SRAD is available in the RSET.
    if (rs_getrad(sys_rset, srad, sradsdl, srad_idx, 0) < 0)
      continue;

    for (int cpu = 0; cpu < max_procs; cpu++) {
      // Set the info for the cpu if it is in the SRAD.
      if (rs_op(RS_TESTRESOURCE, srad, NULL, R_PROCS, cpu)) {
        threadInfo[cpu][osIdIndex] = cpu;
        threadInfo[cpu][pkgIdIndex] = cur_rad;
        threadInfo[cpu][coreIdIndex] = cpu / smt_threads;
        ++num_set;
        if (num_set >= num_avail) {
          // Done if all available CPUs have been set.
          break;
        }
      }
    }
    ++cur_rad;
  }
  rs_free(sys_rset);
  rs_free(srad);

  // The topology is already sorted.

#else // !KMP_OS_AIX
  unsigned num_avail = 0;
  *line = 0;
#if KMP_ARCH_S390X
  bool reading_s390x_sys_info = true;
#endif
  while (!feof(f)) {
    // Create an inner scoping level, so that all the goto targets at the end of
    // the loop appear in an outer scoping level. This avoids warnings about
    // jumping past an initialization to a target in the same block.
    {
      buf[sizeof(buf) - 1] = 1;
      bool long_line = false;
      if (!fgets(buf, sizeof(buf), f)) {
        // Read errors presumably because of EOF
        // If there is valid data in threadInfo[num_avail], then fake
        // a blank line in ensure that the last address gets parsed.
        bool valid = false;
        for (i = 0; i <= maxIndex; i++) {
          if (threadInfo[num_avail][i] != UINT_MAX) {
            valid = true;
          }
        }
        if (!valid) {
          break;
        }
        buf[0] = 0;
      } else if (!buf[sizeof(buf) - 1]) {
        // The line is longer than the buffer.  Set a flag and don't
        // emit an error if we were going to ignore the line, anyway.
        long_line = true;

#define CHECK_LINE                                                             \
  if (long_line) {                                                             \
    CLEANUP_THREAD_INFO;                                                       \
    *msg_id = kmp_i18n_str_LongLineCpuinfo;                                    \
    return false;                                                              \
  }
      }
      (*line)++;

#if KMP_ARCH_LOONGARCH64
      // The parsing logic of /proc/cpuinfo in this function highly depends on
      // the blank lines between each processor info block. But on LoongArch a
      // blank line exists before the first processor info block (i.e. after the
      // "system type" line). This blank line was added because the "system
      // type" line is unrelated to any of the CPUs. We must skip this line so
      // that the original logic works on LoongArch.
      if (*buf == '\n' && *line == 2)
        continue;
#endif
#if KMP_ARCH_S390X
      // s390x /proc/cpuinfo starts with a variable number of lines containing
      // the overall system information. Skip them.
      if (reading_s390x_sys_info) {
        if (*buf == '\n')
          reading_s390x_sys_info = false;
        continue;
      }
#endif

#if KMP_ARCH_S390X
      char s1[] = "cpu number";
#else
      char s1[] = "processor";
#endif
      if (strncmp(buf, s1, sizeof(s1) - 1) == 0) {
        CHECK_LINE;
        char *p = strchr(buf + sizeof(s1) - 1, ':');
        unsigned val;
        if ((p == NULL) || (KMP_SSCANF(p + 1, "%u\n", &val) != 1))
          goto no_val;
        if (threadInfo[num_avail][osIdIndex] != UINT_MAX)
#if KMP_ARCH_AARCH64
          // Handle the old AArch64 /proc/cpuinfo layout differently,
          // it contains all of the 'processor' entries listed in a
          // single 'Processor' section, therefore the normal looking
          // for duplicates in that section will always fail.
          num_avail++;
#else
          goto dup_field;
#endif
        threadInfo[num_avail][osIdIndex] = val;
#if KMP_OS_LINUX && !(KMP_ARCH_X86 || KMP_ARCH_X86_64)
        char path[256];
        KMP_SNPRINTF(
            path, sizeof(path),
            "/sys/devices/system/cpu/cpu%u/topology/physical_package_id",
            threadInfo[num_avail][osIdIndex]);
        __kmp_read_from_file(path, "%u", &threadInfo[num_avail][pkgIdIndex]);

#if KMP_ARCH_S390X
        // Disambiguate physical_package_id.
        unsigned book_id;
        KMP_SNPRINTF(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%u/topology/book_id",
                     threadInfo[num_avail][osIdIndex]);
        __kmp_read_from_file(path, "%u", &book_id);
        threadInfo[num_avail][pkgIdIndex] |= (book_id << 8);

        unsigned drawer_id;
        KMP_SNPRINTF(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%u/topology/drawer_id",
                     threadInfo[num_avail][osIdIndex]);
        __kmp_read_from_file(path, "%u", &drawer_id);
        threadInfo[num_avail][pkgIdIndex] |= (drawer_id << 16);
#endif

        KMP_SNPRINTF(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%u/topology/core_id",
                     threadInfo[num_avail][osIdIndex]);
        __kmp_read_from_file(path, "%u", &threadInfo[num_avail][coreIdIndex]);
        continue;
#else
      }
      char s2[] = "physical id";
      if (strncmp(buf, s2, sizeof(s2) - 1) == 0) {
        CHECK_LINE;
        char *p = strchr(buf + sizeof(s2) - 1, ':');
        unsigned val;
        if ((p == NULL) || (KMP_SSCANF(p + 1, "%u\n", &val) != 1))
          goto no_val;
        if (threadInfo[num_avail][pkgIdIndex] != UINT_MAX)
          goto dup_field;
        threadInfo[num_avail][pkgIdIndex] = val;
        continue;
      }
      char s3[] = "core id";
      if (strncmp(buf, s3, sizeof(s3) - 1) == 0) {
        CHECK_LINE;
        char *p = strchr(buf + sizeof(s3) - 1, ':');
        unsigned val;
        if ((p == NULL) || (KMP_SSCANF(p + 1, "%u\n", &val) != 1))
          goto no_val;
        if (threadInfo[num_avail][coreIdIndex] != UINT_MAX)
          goto dup_field;
        threadInfo[num_avail][coreIdIndex] = val;
        continue;
#endif // KMP_OS_LINUX && USE_SYSFS_INFO
      }
      char s4[] = "thread id";
      if (strncmp(buf, s4, sizeof(s4) - 1) == 0) {
        CHECK_LINE;
        char *p = strchr(buf + sizeof(s4) - 1, ':');
        unsigned val;
        if ((p == NULL) || (KMP_SSCANF(p + 1, "%u\n", &val) != 1))
          goto no_val;
        if (threadInfo[num_avail][threadIdIndex] != UINT_MAX)
          goto dup_field;
        threadInfo[num_avail][threadIdIndex] = val;
        continue;
      }
      unsigned level;
      if (KMP_SSCANF(buf, "node_%u id", &level) == 1) {
        CHECK_LINE;
        char *p = strchr(buf + sizeof(s4) - 1, ':');
        unsigned val;
        if ((p == NULL) || (KMP_SSCANF(p + 1, "%u\n", &val) != 1))
          goto no_val;
        // validate the input before using level:
        if (level > (unsigned)__kmp_xproc) { // level is too big
          level = __kmp_xproc;
        }
        if (threadInfo[num_avail][nodeIdIndex + level] != UINT_MAX)
          goto dup_field;
        threadInfo[num_avail][nodeIdIndex + level] = val;
        continue;
      }

      // We didn't recognize the leading token on the line. There are lots of
      // leading tokens that we don't recognize - if the line isn't empty, go on
      // to the next line.
      if ((*buf != 0) && (*buf != '\n')) {
        // If the line is longer than the buffer, read characters
        // until we find a newline.
        if (long_line) {
          int ch;
          while (((ch = fgetc(f)) != EOF) && (ch != '\n'))
            ;
        }
        continue;
      }

      // A newline has signalled the end of the processor record.
      // Check that there aren't too many procs specified.
      if ((int)num_avail == __kmp_xproc) {
        CLEANUP_THREAD_INFO;
        *msg_id = kmp_i18n_str_TooManyEntries;
        return false;
      }

      // Check for missing fields.  The osId field must be there, and we
      // currently require that the physical id field is specified, also.
      if (threadInfo[num_avail][osIdIndex] == UINT_MAX) {
        CLEANUP_THREAD_INFO;
        *msg_id = kmp_i18n_str_MissingProcField;
        return false;
      }
      if (threadInfo[0][pkgIdIndex] == UINT_MAX) {
        CLEANUP_THREAD_INFO;
        *msg_id = kmp_i18n_str_MissingPhysicalIDField;
        return false;
      }

      // Skip this proc if it is not included in the machine model.
      if (KMP_AFFINITY_CAPABLE() &&
          !KMP_CPU_ISSET(threadInfo[num_avail][osIdIndex],
                         __kmp_affin_fullMask)) {
        INIT_PROC_INFO(threadInfo[num_avail]);
        continue;
      }

      // We have a successful parse of this proc's info.
      // Increment the counter, and prepare for the next proc.
      num_avail++;
      KMP_ASSERT(num_avail <= num_records);
      INIT_PROC_INFO(threadInfo[num_avail]);
    }
    continue;

  no_val:
    CLEANUP_THREAD_INFO;
    *msg_id = kmp_i18n_str_MissingValCpuinfo;
    return false;

  dup_field:
    CLEANUP_THREAD_INFO;
    *msg_id = kmp_i18n_str_DuplicateFieldCpuinfo;
    return false;
  }
  *line = 0;

#if KMP_MIC && REDUCE_TEAM_SIZE
  unsigned teamSize = 0;
#endif // KMP_MIC && REDUCE_TEAM_SIZE

  // check for num_records == __kmp_xproc ???

  // If it is configured to omit the package level when there is only a single
  // package, the logic at the end of this routine won't work if there is only a
  // single thread
  KMP_ASSERT(num_avail > 0);
  KMP_ASSERT(num_avail <= num_records);

  // Sort the threadInfo table by physical Id.
  qsort(threadInfo, num_avail, sizeof(*threadInfo),
        __kmp_affinity_cmp_ProcCpuInfo_phys_id);

#endif // KMP_OS_AIX

  // The table is now sorted by pkgId / coreId / threadId, but we really don't
  // know the radix of any of the fields. pkgId's may be sparsely assigned among
  // the chips on a system. Although coreId's are usually assigned
  // [0 .. coresPerPkg-1] and threadId's are usually assigned
  // [0..threadsPerCore-1], we don't want to make any such assumptions.
  //
  // For that matter, we don't know what coresPerPkg and threadsPerCore (or the
  // total # packages) are at this point - we want to determine that now. We
  // only have an upper bound on the first two figures.
  unsigned *counts =
      (unsigned *)__kmp_allocate((maxIndex + 1) * sizeof(unsigned));
  unsigned *maxCt =
      (unsigned *)__kmp_allocate((maxIndex + 1) * sizeof(unsigned));
  unsigned *totals =
      (unsigned *)__kmp_allocate((maxIndex + 1) * sizeof(unsigned));
  unsigned *lastId =
      (unsigned *)__kmp_allocate((maxIndex + 1) * sizeof(unsigned));

  bool assign_thread_ids = false;
  unsigned threadIdCt;
  unsigned index;

restart_radix_check:
  threadIdCt = 0;

  // Initialize the counter arrays with data from threadInfo[0].
  if (assign_thread_ids) {
    if (threadInfo[0][threadIdIndex] == UINT_MAX) {
      threadInfo[0][threadIdIndex] = threadIdCt++;
    } else if (threadIdCt <= threadInfo[0][threadIdIndex]) {
      threadIdCt = threadInfo[0][threadIdIndex] + 1;
    }
  }
  for (index = 0; index <= maxIndex; index++) {
    counts[index] = 1;
    maxCt[index] = 1;
    totals[index] = 1;
    lastId[index] = threadInfo[0][index];
    ;
  }

  // Run through the rest of the OS procs.
  for (i = 1; i < num_avail; i++) {
    // Find the most significant index whose id differs from the id for the
    // previous OS proc.
    for (index = maxIndex; index >= threadIdIndex; index--) {
      if (assign_thread_ids && (index == threadIdIndex)) {
        // Auto-assign the thread id field if it wasn't specified.
        if (threadInfo[i][threadIdIndex] == UINT_MAX) {
          threadInfo[i][threadIdIndex] = threadIdCt++;
        }
        // Apparently the thread id field was specified for some entries and not
        // others. Start the thread id counter off at the next higher thread id.
        else if (threadIdCt <= threadInfo[i][threadIdIndex]) {
          threadIdCt = threadInfo[i][threadIdIndex] + 1;
        }
      }
      if (threadInfo[i][index] != lastId[index]) {
        // Run through all indices which are less significant, and reset the
        // counts to 1. At all levels up to and including index, we need to
        // increment the totals and record the last id.
        unsigned index2;
        for (index2 = threadIdIndex; index2 < index; index2++) {
          totals[index2]++;
          if (counts[index2] > maxCt[index2]) {
            maxCt[index2] = counts[index2];
          }
          counts[index2] = 1;
          lastId[index2] = threadInfo[i][index2];
        }
        counts[index]++;
        totals[index]++;
        lastId[index] = threadInfo[i][index];

        if (assign_thread_ids && (index > threadIdIndex)) {

#if KMP_MIC && REDUCE_TEAM_SIZE
          // The default team size is the total #threads in the machine
          // minus 1 thread for every core that has 3 or more threads.
          teamSize += (threadIdCt <= 2) ? (threadIdCt) : (threadIdCt - 1);
#endif // KMP_MIC && REDUCE_TEAM_SIZE

          // Restart the thread counter, as we are on a new core.
          threadIdCt = 0;

          // Auto-assign the thread id field if it wasn't specified.
          if (threadInfo[i][threadIdIndex] == UINT_MAX) {
            threadInfo[i][threadIdIndex] = threadIdCt++;
          }

          // Apparently the thread id field was specified for some entries and
          // not others. Start the thread id counter off at the next higher
          // thread id.
          else if (threadIdCt <= threadInfo[i][threadIdIndex]) {
            threadIdCt = threadInfo[i][threadIdIndex] + 1;
          }
        }
        break;
      }
    }
    if (index < threadIdIndex) {
      // If thread ids were specified, it is an error if they are not unique.
      // Also, check that we waven't already restarted the loop (to be safe -
      // shouldn't need to).
      if ((threadInfo[i][threadIdIndex] != UINT_MAX) || assign_thread_ids) {
        __kmp_free(lastId);
        __kmp_free(totals);
        __kmp_free(maxCt);
        __kmp_free(counts);
        CLEANUP_THREAD_INFO;
        *msg_id = kmp_i18n_str_PhysicalIDsNotUnique;
        return false;
      }

      // If the thread ids were not specified and we see entries that
      // are duplicates, start the loop over and assign the thread ids manually.
      assign_thread_ids = true;
      goto restart_radix_check;
    }
  }

#if KMP_MIC && REDUCE_TEAM_SIZE
  // The default team size is the total #threads in the machine
  // minus 1 thread for every core that has 3 or more threads.
  teamSize += (threadIdCt <= 2) ? (threadIdCt) : (threadIdCt - 1);
#endif // KMP_MIC && REDUCE_TEAM_SIZE

  for (index = threadIdIndex; index <= maxIndex; index++) {
    if (counts[index] > maxCt[index]) {
      maxCt[index] = counts[index];
    }
  }

  __kmp_nThreadsPerCore = maxCt[threadIdIndex];
  nCoresPerPkg = maxCt[coreIdIndex];
  nPackages = totals[pkgIdIndex];

  // When affinity is off, this routine will still be called to set
  // __kmp_ncores, as well as __kmp_nThreadsPerCore, nCoresPerPkg, & nPackages.
  // Make sure all these vars are set correctly, and return now if affinity is
  // not enabled.
  __kmp_ncores = totals[coreIdIndex];
  if (!KMP_AFFINITY_CAPABLE()) {
    KMP_ASSERT(__kmp_affinity.type == affinity_none);
    return true;
  }

#if KMP_MIC && REDUCE_TEAM_SIZE
  // Set the default team size.
  if ((__kmp_dflt_team_nth == 0) && (teamSize > 0)) {
    __kmp_dflt_team_nth = teamSize;
    KA_TRACE(20, ("__kmp_affinity_create_cpuinfo_map: setting "
                  "__kmp_dflt_team_nth = %d\n",
                  __kmp_dflt_team_nth));
  }
#endif // KMP_MIC && REDUCE_TEAM_SIZE

  KMP_DEBUG_ASSERT(num_avail == (unsigned)__kmp_avail_proc);

  // Count the number of levels which have more nodes at that level than at the
  // parent's level (with there being an implicit root node of the top level).
  // This is equivalent to saying that there is at least one node at this level
  // which has a sibling. These levels are in the map, and the package level is
  // always in the map.
  bool *inMap = (bool *)__kmp_allocate((maxIndex + 1) * sizeof(bool));
  for (index = threadIdIndex; index < maxIndex; index++) {
    KMP_ASSERT(totals[index] >= totals[index + 1]);
    inMap[index] = (totals[index] > totals[index + 1]);
  }
  inMap[maxIndex] = (totals[maxIndex] > 1);
  inMap[pkgIdIndex] = true;
  inMap[coreIdIndex] = true;
  inMap[threadIdIndex] = true;

  int depth = 0;
  int idx = 0;
  kmp_hw_t types[KMP_HW_LAST];
  int pkgLevel = -1;
  int coreLevel = -1;
  int threadLevel = -1;
  for (index = threadIdIndex; index <= maxIndex; index++) {
    if (inMap[index]) {
      depth++;
    }
  }
  if (inMap[pkgIdIndex]) {
    pkgLevel = idx;
    types[idx++] = KMP_HW_SOCKET;
  }
  if (inMap[coreIdIndex]) {
    coreLevel = idx;
    types[idx++] = KMP_HW_CORE;
  }
  if (inMap[threadIdIndex]) {
    threadLevel = idx;
    types[idx++] = KMP_HW_THREAD;
  }
  KMP_ASSERT(depth > 0);

  // Construct the data structure that is to be returned.
  __kmp_topology = kmp_topology_t::allocate(num_avail, depth, types);

  for (i = 0; i < num_avail; ++i) {
    unsigned os = threadInfo[i][osIdIndex];
    int src_index;
    kmp_hw_thread_t &hw_thread = __kmp_topology->at(i);
    hw_thread.clear();
    hw_thread.os_id = os;

    idx = 0;
    for (src_index = maxIndex; src_index >= threadIdIndex; src_index--) {
      if (!inMap[src_index]) {
        continue;
      }
      if (src_index == pkgIdIndex) {
        hw_thread.ids[pkgLevel] = threadInfo[i][src_index];
      } else if (src_index == coreIdIndex) {
        hw_thread.ids[coreLevel] = threadInfo[i][src_index];
      } else if (src_index == threadIdIndex) {
        hw_thread.ids[threadLevel] = threadInfo[i][src_index];
      }
    }
  }

  __kmp_free(inMap);
  __kmp_free(lastId);
  __kmp_free(totals);
  __kmp_free(maxCt);
  __kmp_free(counts);
  CLEANUP_THREAD_INFO;
  __kmp_topology->sort_ids();
  if (!__kmp_topology->check_ids()) {
    kmp_topology_t::deallocate(__kmp_topology);
    __kmp_topology = nullptr;
    *msg_id = kmp_i18n_str_PhysicalIDsNotUnique;
    return false;
  }
  return true;
}

// Create and return a table of affinity masks, indexed by OS thread ID.
// This routine handles OR'ing together all the affinity masks of threads
// that are sufficiently close, if granularity > fine.
template <typename FindNextFunctionType>
static void __kmp_create_os_id_masks(unsigned *numUnique,
                                     kmp_affinity_t &affinity,
                                     FindNextFunctionType find_next) {
  // First form a table of affinity masks in order of OS thread id.
  int maxOsId;
  int i;
  int numAddrs = __kmp_topology->get_num_hw_threads();
  int depth = __kmp_topology->get_depth();
  const char *env_var = __kmp_get_affinity_env_var(affinity);
  KMP_ASSERT(numAddrs);
  KMP_ASSERT(depth);

  i = find_next(-1);
  // If could not find HW thread location with attributes, then return and
  // fallback to increment find_next and disregard core attributes.
  if (i >= numAddrs)
    return;

  maxOsId = 0;
  for (i = numAddrs - 1;; --i) {
    int osId = __kmp_topology->at(i).os_id;
    if (osId > maxOsId) {
      maxOsId = osId;
    }
    if (i == 0)
      break;
  }
  affinity.num_os_id_masks = maxOsId + 1;
  KMP_CPU_ALLOC_ARRAY(affinity.os_id_masks, affinity.num_os_id_masks);
  KMP_ASSERT(affinity.gran_levels >= 0);
  if (affinity.flags.verbose && (affinity.gran_levels > 0)) {
    KMP_INFORM(ThreadsMigrate, env_var, affinity.gran_levels);
  }
  if (affinity.gran_levels >= (int)depth) {
    KMP_AFF_WARNING(affinity, AffThreadsMayMigrate);
  }

  // Run through the table, forming the masks for all threads on each core.
  // Threads on the same core will have identical kmp_hw_thread_t objects, not
  // considering the last level, which must be the thread id. All threads on a
  // core will appear consecutively.
  int unique = 0;
  int j = 0; // index of 1st thread on core
  int leader = 0;
  kmp_affin_mask_t *sum;
  KMP_CPU_ALLOC_ON_STACK(sum);
  KMP_CPU_ZERO(sum);

  i = j = leader = find_next(-1);
  KMP_CPU_SET(__kmp_topology->at(i).os_id, sum);
  kmp_full_mask_modifier_t full_mask;
  for (i = find_next(i); i < numAddrs; i = find_next(i)) {
    // If this thread is sufficiently close to the leader (within the
    // granularity setting), then set the bit for this os thread in the
    // affinity mask for this group, and go on to the next thread.
    if (__kmp_topology->is_close(leader, i, affinity)) {
      KMP_CPU_SET(__kmp_topology->at(i).os_id, sum);
      continue;
    }

    // For every thread in this group, copy the mask to the thread's entry in
    // the OS Id mask table. Mark the first address as a leader.
    for (; j < i; j = find_next(j)) {
      int osId = __kmp_topology->at(j).os_id;
      KMP_DEBUG_ASSERT(osId <= maxOsId);
      kmp_affin_mask_t *mask = KMP_CPU_INDEX(affinity.os_id_masks, osId);
      KMP_CPU_COPY(mask, sum);
      __kmp_topology->at(j).leader = (j == leader);
    }
    unique++;

    // Start a new mask.
    leader = i;
    full_mask.include(sum);
    KMP_CPU_ZERO(sum);
    KMP_CPU_SET(__kmp_topology->at(i).os_id, sum);
  }

  // For every thread in last group, copy the mask to the thread's
  // entry in the OS Id mask table.
  for (; j < i; j = find_next(j)) {
    int osId = __kmp_topology->at(j).os_id;
    KMP_DEBUG_ASSERT(osId <= maxOsId);
    kmp_affin_mask_t *mask = KMP_CPU_INDEX(affinity.os_id_masks, osId);
    KMP_CPU_COPY(mask, sum);
    __kmp_topology->at(j).leader = (j == leader);
  }
  full_mask.include(sum);
  unique++;
  KMP_CPU_FREE_FROM_STACK(sum);

  // See if the OS Id mask table further restricts or changes the full mask
  if (full_mask.restrict_to_mask() && affinity.flags.verbose) {
    __kmp_topology->print(env_var);
  }

  *numUnique = unique;
}

// Stuff for the affinity proclist parsers.  It's easier to declare these vars
// as file-static than to try and pass them through the calling sequence of
// the recursive-descent OMP_PLACES parser.
static kmp_affin_mask_t *newMasks;
static int numNewMasks;
static int nextNewMask;

#define ADD_MASK(_mask)                                                        \
  {                                                                            \
    if (nextNewMask >= numNewMasks) {                                          \
      int i;                                                                   \
      numNewMasks *= 2;                                                        \
      kmp_affin_mask_t *temp;                                                  \
      KMP_CPU_INTERNAL_ALLOC_ARRAY(temp, numNewMasks);                         \
      for (i = 0; i < numNewMasks / 2; i++) {                                  \
        kmp_affin_mask_t *src = KMP_CPU_INDEX(newMasks, i);                    \
        kmp_affin_mask_t *dest = KMP_CPU_INDEX(temp, i);                       \
        KMP_CPU_COPY(dest, src);                                               \
      }                                                                        \
      KMP_CPU_INTERNAL_FREE_ARRAY(newMasks, numNewMasks / 2);                  \
      newMasks = temp;                                                         \
    }                                                                          \
    KMP_CPU_COPY(KMP_CPU_INDEX(newMasks, nextNewMask), (_mask));               \
    nextNewMask++;                                                             \
  }

#define ADD_MASK_OSID(_osId, _osId2Mask, _maxOsId)                             \
  {                                                                            \
    if (((_osId) > _maxOsId) ||                                                \
        (!KMP_CPU_ISSET((_osId), KMP_CPU_INDEX((_osId2Mask), (_osId))))) {     \
      KMP_AFF_WARNING(affinity, AffIgnoreInvalidProcID, _osId);                \
    } else {                                                                   \
      ADD_MASK(KMP_CPU_INDEX(_osId2Mask, (_osId)));                            \
    }                                                                          \
  }

// Re-parse the proclist (for the explicit affinity type), and form the list
// of affinity newMasks indexed by gtid.
static void __kmp_affinity_process_proclist(kmp_affinity_t &affinity) {
  int i;
  kmp_affin_mask_t **out_masks = &affinity.masks;
  unsigned *out_numMasks = &affinity.num_masks;
  const char *proclist = affinity.proclist;
  kmp_affin_mask_t *osId2Mask = affinity.os_id_masks;
  int maxOsId = affinity.num_os_id_masks - 1;
  const char *scan = proclist;
  const char *next = proclist;

  // We use malloc() for the temporary mask vector, so that we can use
  // realloc() to extend it.
  numNewMasks = 2;
  KMP_CPU_INTERNAL_ALLOC_ARRAY(newMasks, numNewMasks);
  nextNewMask = 0;
  kmp_affin_mask_t *sumMask;
  KMP_CPU_ALLOC(sumMask);
  int setSize = 0;

  for (;;) {
    int start, end, stride;

    SKIP_WS(scan);
    next = scan;
    if (*next == '\0') {
      break;
    }

    if (*next == '{') {
      int num;
      setSize = 0;
      next++; // skip '{'
      SKIP_WS(next);
      scan = next;

      // Read the first integer in the set.
      KMP_ASSERT2((*next >= '0') && (*next <= '9'), "bad proclist");
      SKIP_DIGITS(next);
      num = __kmp_str_to_int(scan, *next);
      KMP_ASSERT2(num >= 0, "bad explicit proc list");

      // Copy the mask for that osId to the sum (union) mask.
      if ((num > maxOsId) ||
          (!KMP_CPU_ISSET(num, KMP_CPU_INDEX(osId2Mask, num)))) {
        KMP_AFF_WARNING(affinity, AffIgnoreInvalidProcID, num);
        KMP_CPU_ZERO(sumMask);
      } else {
        KMP_CPU_COPY(sumMask, KMP_CPU_INDEX(osId2Mask, num));
        setSize = 1;
      }

      for (;;) {
        // Check for end of set.
        SKIP_WS(next);
        if (*next == '}') {
          next++; // skip '}'
          break;
        }

        // Skip optional comma.
        if (*next == ',') {
          next++;
        }
        SKIP_WS(next);

        // Read the next integer in the set.
        scan = next;
        KMP_ASSERT2((*next >= '0') && (*next <= '9'), "bad explicit proc list");

        SKIP_DIGITS(next);
        num = __kmp_str_to_int(scan, *next);
        KMP_ASSERT2(num >= 0, "bad explicit proc list");

        // Add the mask for that osId to the sum mask.
        if ((num > maxOsId) ||
            (!KMP_CPU_ISSET(num, KMP_CPU_INDEX(osId2Mask, num)))) {
          KMP_AFF_WARNING(affinity, AffIgnoreInvalidProcID, num);
        } else {
          KMP_CPU_UNION(sumMask, KMP_CPU_INDEX(osId2Mask, num));
          setSize++;
        }
      }
      if (setSize > 0) {
        ADD_MASK(sumMask);
      }

      SKIP_WS(next);
      if (*next == ',') {
        next++;
      }
      scan = next;
      continue;
    }

    // Read the first integer.
    KMP_ASSERT2((*next >= '0') && (*next <= '9'), "bad explicit proc list");
    SKIP_DIGITS(next);
    start = __kmp_str_to_int(scan, *next);
    KMP_ASSERT2(start >= 0, "bad explicit proc list");
    SKIP_WS(next);

    // If this isn't a range, then add a mask to the list and go on.
    if (*next != '-') {
      ADD_MASK_OSID(start, osId2Mask, maxOsId);

      // Skip optional comma.
      if (*next == ',') {
        next++;
      }
      scan = next;
      continue;
    }

    // This is a range.  Skip over the '-' and read in the 2nd int.
    next++; // skip '-'
    SKIP_WS(next);
    scan = next;
    KMP_ASSERT2((*next >= '0') && (*next <= '9'), "bad explicit proc list");
    SKIP_DIGITS(next);
    end = __kmp_str_to_int(scan, *next);
    KMP_ASSERT2(end >= 0, "bad explicit proc list");

    // Check for a stride parameter
    stride = 1;
    SKIP_WS(next);
    if (*next == ':') {
      // A stride is specified.  Skip over the ':" and read the 3rd int.
      int sign = +1;
      next++; // skip ':'
      SKIP_WS(next);
      scan = next;
      if (*next == '-') {
        sign = -1;
        next++;
        SKIP_WS(next);
        scan = next;
      }
      KMP_ASSERT2((*next >= '0') && (*next <= '9'), "bad explicit proc list");
      SKIP_DIGITS(next);
      stride = __kmp_str_to_int(scan, *next);
      KMP_ASSERT2(stride >= 0, "bad explicit proc list");
      stride *= sign;
    }

    // Do some range checks.
    KMP_ASSERT2(stride != 0, "bad explicit proc list");
    if (stride > 0) {
      KMP_ASSERT2(start <= end, "bad explicit proc list");
    } else {
      KMP_ASSERT2(start >= end, "bad explicit proc list");
    }
    KMP_ASSERT2((end - start) / stride <= 65536, "bad explicit proc list");

    // Add the mask for each OS proc # to the list.
    if (stride > 0) {
      do {
        ADD_MASK_OSID(start, osId2Mask, maxOsId);
        start += stride;
      } while (start <= end);
    } else {
      do {
        ADD_MASK_OSID(start, osId2Mask, maxOsId);
        start += stride;
      } while (start >= end);
    }

    // Skip optional comma.
    SKIP_WS(next);
    if (*next == ',') {
      next++;
    }
    scan = next;
  }

  *out_numMasks = nextNewMask;
  if (nextNewMask == 0) {
    *out_masks = NULL;
    KMP_CPU_INTERNAL_FREE_ARRAY(newMasks, numNewMasks);
    return;
  }
  KMP_CPU_ALLOC_ARRAY((*out_masks), nextNewMask);
  for (i = 0; i < nextNewMask; i++) {
    kmp_affin_mask_t *src = KMP_CPU_INDEX(newMasks, i);
    kmp_affin_mask_t *dest = KMP_CPU_INDEX((*out_masks), i);
    KMP_CPU_COPY(dest, src);
  }
  KMP_CPU_INTERNAL_FREE_ARRAY(newMasks, numNewMasks);
  KMP_CPU_FREE(sumMask);
}

/*-----------------------------------------------------------------------------
Re-parse the OMP_PLACES proc id list, forming the newMasks for the different
places.  Again, Here is the grammar:

place_list := place
place_list := place , place_list
place := num
place := place : num
place := place : num : signed
place := { subplacelist }
place := ! place                  // (lowest priority)
subplace_list := subplace
subplace_list := subplace , subplace_list
subplace := num
subplace := num : num
subplace := num : num : signed
signed := num
signed := + signed
signed := - signed
-----------------------------------------------------------------------------*/
static void __kmp_process_subplace_list(const char **scan,
                                        kmp_affinity_t &affinity, int maxOsId,
                                        kmp_affin_mask_t *tempMask,
                                        int *setSize) {
  const char *next;
  kmp_affin_mask_t *osId2Mask = affinity.os_id_masks;

  for (;;) {
    int start, count, stride, i;

    // Read in the starting proc id
    SKIP_WS(*scan);
    KMP_ASSERT2((**scan >= '0') && (**scan <= '9'), "bad explicit places list");
    next = *scan;
    SKIP_DIGITS(next);
    start = __kmp_str_to_int(*scan, *next);
    KMP_ASSERT(start >= 0);
    *scan = next;

    // valid follow sets are ',' ':' and '}'
    SKIP_WS(*scan);
    if (**scan == '}' || **scan == ',') {
      if ((start > maxOsId) ||
          (!KMP_CPU_ISSET(start, KMP_CPU_INDEX(osId2Mask, start)))) {
        KMP_AFF_WARNING(affinity, AffIgnoreInvalidProcID, start);
      } else {
        KMP_CPU_UNION(tempMask, KMP_CPU_INDEX(osId2Mask, start));
        (*setSize)++;
      }
      if (**scan == '}') {
        break;
      }
      (*scan)++; // skip ','
      continue;
    }
    KMP_ASSERT2(**scan == ':', "bad explicit places list");
    (*scan)++; // skip ':'

    // Read count parameter
    SKIP_WS(*scan);
    KMP_ASSERT2((**scan >= '0') && (**scan <= '9'), "bad explicit places list");
    next = *scan;
    SKIP_DIGITS(next);
    count = __kmp_str_to_int(*scan, *next);
    KMP_ASSERT(count >= 0);
    *scan = next;

    // valid follow sets are ',' ':' and '}'
    SKIP_WS(*scan);
    if (**scan == '}' || **scan == ',') {
      for (i = 0; i < count; i++) {
        if ((start > maxOsId) ||
            (!KMP_CPU_ISSET(start, KMP_CPU_INDEX(osId2Mask, start)))) {
          KMP_AFF_WARNING(affinity, AffIgnoreInvalidProcID, start);
          break; // don't proliferate warnings for large count
        } else {
          KMP_CPU_UNION(tempMask, KMP_CPU_INDEX(osId2Mask, start));
          start++;
          (*setSize)++;
        }
      }
      if (**scan == '}') {
        break;
      }
      (*scan)++; // skip ','
      continue;
    }
    KMP_ASSERT2(**scan == ':', "bad explicit places list");
    (*scan)++; // skip ':'

    // Read stride parameter
    int sign = +1;
    for (;;) {
      SKIP_WS(*scan);
      if (**scan == '+') {
        (*scan)++; // skip '+'
        continue;
      }
      if (**scan == '-') {
        sign *= -1;
        (*scan)++; // skip '-'
        continue;
      }
      break;
    }
    SKIP_WS(*scan);
    KMP_ASSERT2((**scan >= '0') && (**scan <= '9'), "bad explicit places list");
    next = *scan;
    SKIP_DIGITS(next);
    stride = __kmp_str_to_int(*scan, *next);
    KMP_ASSERT(stride >= 0);
    *scan = next;
    stride *= sign;

    // valid follow sets are ',' and '}'
    SKIP_WS(*scan);
    if (**scan == '}' || **scan == ',') {
      for (i = 0; i < count; i++) {
        if ((start > maxOsId) ||
            (!KMP_CPU_ISSET(start, KMP_CPU_INDEX(osId2Mask, start)))) {
          KMP_AFF_WARNING(affinity, AffIgnoreInvalidProcID, start);
          break; // don't proliferate warnings for large count
        } else {
          KMP_CPU_UNION(tempMask, KMP_CPU_INDEX(osId2Mask, start));
          start += stride;
          (*setSize)++;
        }
      }
      if (**scan == '}') {
        break;
      }
      (*scan)++; // skip ','
      continue;
    }

    KMP_ASSERT2(0, "bad explicit places list");
  }
}

static void __kmp_process_place(const char **scan, kmp_affinity_t &affinity,
                                int maxOsId, kmp_affin_mask_t *tempMask,
                                int *setSize) {
  const char *next;
  kmp_affin_mask_t *osId2Mask = affinity.os_id_masks;

  // valid follow sets are '{' '!' and num
  SKIP_WS(*scan);
  if (**scan == '{') {
    (*scan)++; // skip '{'
    __kmp_process_subplace_list(scan, affinity, maxOsId, tempMask, setSize);
    KMP_ASSERT2(**scan == '}', "bad explicit places list");
    (*scan)++; // skip '}'
  } else if (**scan == '!') {
    (*scan)++; // skip '!'
    __kmp_process_place(scan, affinity, maxOsId, tempMask, setSize);
    KMP_CPU_COMPLEMENT(maxOsId, tempMask);
  } else if ((**scan >= '0') && (**scan <= '9')) {
    next = *scan;
    SKIP_DIGITS(next);
    int num = __kmp_str_to_int(*scan, *next);
    KMP_ASSERT(num >= 0);
    if ((num > maxOsId) ||
        (!KMP_CPU_ISSET(num, KMP_CPU_INDEX(osId2Mask, num)))) {
      KMP_AFF_WARNING(affinity, AffIgnoreInvalidProcID, num);
    } else {
      KMP_CPU_UNION(tempMask, KMP_CPU_INDEX(osId2Mask, num));
      (*setSize)++;
    }
    *scan = next; // skip num
  } else {
    KMP_ASSERT2(0, "bad explicit places list");
  }
}

// static void
void __kmp_affinity_process_placelist(kmp_affinity_t &affinity) {
  int i, j, count, stride, sign;
  kmp_affin_mask_t **out_masks = &affinity.masks;
  unsigned *out_numMasks = &affinity.num_masks;
  const char *placelist = affinity.proclist;
  kmp_affin_mask_t *osId2Mask = affinity.os_id_masks;
  int maxOsId = affinity.num_os_id_masks - 1;
  const char *scan = placelist;
  const char *next = placelist;

  numNewMasks = 2;
  KMP_CPU_INTERNAL_ALLOC_ARRAY(newMasks, numNewMasks);
  nextNewMask = 0;

  // tempMask is modified based on the previous or initial
  //   place to form the current place
  // previousMask contains the previous place
  kmp_affin_mask_t *tempMask;
  kmp_affin_mask_t *previousMask;
  KMP_CPU_ALLOC(tempMask);
  KMP_CPU_ZERO(tempMask);
  KMP_CPU_ALLOC(previousMask);
  KMP_CPU_ZERO(previousMask);
  int setSize = 0;

  for (;;) {
    __kmp_process_place(&scan, affinity, maxOsId, tempMask, &setSize);

    // valid follow sets are ',' ':' and EOL
    SKIP_WS(scan);
    if (*scan == '\0' || *scan == ',') {
      if (setSize > 0) {
        ADD_MASK(tempMask);
      }
      KMP_CPU_ZERO(tempMask);
      setSize = 0;
      if (*scan == '\0') {
        break;
      }
      scan++; // skip ','
      continue;
    }

    KMP_ASSERT2(*scan == ':', "bad explicit places list");
    scan++; // skip ':'

    // Read count parameter
    SKIP_WS(scan);
    KMP_ASSERT2((*scan >= '0') && (*scan <= '9'), "bad explicit places list");
    next = scan;
    SKIP_DIGITS(next);
    count = __kmp_str_to_int(scan, *next);
    KMP_ASSERT(count >= 0);
    scan = next;

    // valid follow sets are ',' ':' and EOL
    SKIP_WS(scan);
    if (*scan == '\0' || *scan == ',') {
      stride = +1;
    } else {
      KMP_ASSERT2(*scan == ':', "bad explicit places list");
      scan++; // skip ':'

      // Read stride parameter
      sign = +1;
      for (;;) {
        SKIP_WS(scan);
        if (*scan == '+') {
          scan++; // skip '+'
          continue;
        }
        if (*scan == '-') {
          sign *= -1;
          scan++; // skip '-'
          continue;
        }
        break;
      }
      SKIP_WS(scan);
      KMP_ASSERT2((*scan >= '0') && (*scan <= '9'), "bad explicit places list");
      next = scan;
      SKIP_DIGITS(next);
      stride = __kmp_str_to_int(scan, *next);
      KMP_DEBUG_ASSERT(stride >= 0);
      scan = next;
      stride *= sign;
    }

    // Add places determined by initial_place : count : stride
    for (i = 0; i < count; i++) {
      if (setSize == 0) {
        break;
      }
      // Add the current place, then build the next place (tempMask) from that
      KMP_CPU_COPY(previousMask, tempMask);
      ADD_MASK(previousMask);
      KMP_CPU_ZERO(tempMask);
      setSize = 0;
      KMP_CPU_SET_ITERATE(j, previousMask) {
        if (!KMP_CPU_ISSET(j, previousMask)) {
          continue;
        }
        if ((j + stride > maxOsId) || (j + stride < 0) ||
            (!KMP_CPU_ISSET(j, __kmp_affin_fullMask)) ||
            (!KMP_CPU_ISSET(j + stride,
                            KMP_CPU_INDEX(osId2Mask, j + stride)))) {
          if (i < count - 1) {
            KMP_AFF_WARNING(affinity, AffIgnoreInvalidProcID, j + stride);
          }
          continue;
        }
        KMP_CPU_SET(j + stride, tempMask);
        setSize++;
      }
    }
    KMP_CPU_ZERO(tempMask);
    setSize = 0;

    // valid follow sets are ',' and EOL
    SKIP_WS(scan);
    if (*scan == '\0') {
      break;
    }
    if (*scan == ',') {
      scan++; // skip ','
      continue;
    }

    KMP_ASSERT2(0, "bad explicit places list");
  }

  *out_numMasks = nextNewMask;
  if (nextNewMask == 0) {
    *out_masks = NULL;
    KMP_CPU_INTERNAL_FREE_ARRAY(newMasks, numNewMasks);
    return;
  }
  KMP_CPU_ALLOC_ARRAY((*out_masks), nextNewMask);
  KMP_CPU_FREE(tempMask);
  KMP_CPU_FREE(previousMask);
  for (i = 0; i < nextNewMask; i++) {
    kmp_affin_mask_t *src = KMP_CPU_INDEX(newMasks, i);
    kmp_affin_mask_t *dest = KMP_CPU_INDEX((*out_masks), i);
    KMP_CPU_COPY(dest, src);
  }
  KMP_CPU_INTERNAL_FREE_ARRAY(newMasks, numNewMasks);
}

#undef ADD_MASK
#undef ADD_MASK_OSID

// This function figures out the deepest level at which there is at least one
// cluster/core with more than one processing unit bound to it.
static int __kmp_affinity_find_core_level(int nprocs, int bottom_level) {
  int core_level = 0;

  for (int i = 0; i < nprocs; i++) {
    const kmp_hw_thread_t &hw_thread = __kmp_topology->at(i);
    for (int j = bottom_level; j > 0; j--) {
      if (hw_thread.ids[j] > 0) {
        if (core_level < (j - 1)) {
          core_level = j - 1;
        }
      }
    }
  }
  return core_level;
}

// This function counts number of clusters/cores at given level.
static int __kmp_affinity_compute_ncores(int nprocs, int bottom_level,
                                         int core_level) {
  return __kmp_topology->get_count(core_level);
}
// This function finds to which cluster/core given processing unit is bound.
static int __kmp_affinity_find_core(int proc, int bottom_level,
                                    int core_level) {
  int core = 0;
  KMP_DEBUG_ASSERT(proc >= 0 && proc < __kmp_topology->get_num_hw_threads());
  for (int i = 0; i <= proc; ++i) {
    if (i + 1 <= proc) {
      for (int j = 0; j <= core_level; ++j) {
        if (__kmp_topology->at(i + 1).sub_ids[j] !=
            __kmp_topology->at(i).sub_ids[j]) {
          core++;
          break;
        }
      }
    }
  }
  return core;
}

// This function finds maximal number of processing units bound to a
// cluster/core at given level.
static int __kmp_affinity_max_proc_per_core(int nprocs, int bottom_level,
                                            int core_level) {
  if (core_level >= bottom_level)
    return 1;
  int thread_level = __kmp_topology->get_level(KMP_HW_THREAD);
  return __kmp_topology->calculate_ratio(thread_level, core_level);
}

static int *procarr = NULL;
static int __kmp_aff_depth = 0;
static int *__kmp_osid_to_hwthread_map = NULL;

static void __kmp_affinity_get_mask_topology_info(const kmp_affin_mask_t *mask,
                                                  kmp_affinity_ids_t &ids,
                                                  kmp_affinity_attrs_t &attrs) {
  if (!KMP_AFFINITY_CAPABLE())
    return;

  // Initiailze ids and attrs thread data
  for (int i = 0; i < KMP_HW_LAST; ++i)
    ids.ids[i] = kmp_hw_thread_t::UNKNOWN_ID;
  attrs = KMP_AFFINITY_ATTRS_UNKNOWN;

  // Iterate through each os id within the mask and determine
  // the topology id and attribute information
  int cpu;
  int depth = __kmp_topology->get_depth();
  KMP_CPU_SET_ITERATE(cpu, mask) {
    int osid_idx = __kmp_osid_to_hwthread_map[cpu];
    ids.os_id = cpu;
    const kmp_hw_thread_t &hw_thread = __kmp_topology->at(osid_idx);
    for (int level = 0; level < depth; ++level) {
      kmp_hw_t type = __kmp_topology->get_type(level);
      int id = hw_thread.sub_ids[level];
      if (ids.ids[type] == kmp_hw_thread_t::UNKNOWN_ID || ids.ids[type] == id) {
        ids.ids[type] = id;
      } else {
        // This mask spans across multiple topology units, set it as such
        // and mark every level below as such as well.
        ids.ids[type] = kmp_hw_thread_t::MULTIPLE_ID;
        for (; level < depth; ++level) {
          kmp_hw_t type = __kmp_topology->get_type(level);
          ids.ids[type] = kmp_hw_thread_t::MULTIPLE_ID;
        }
      }
    }
    if (!attrs.valid) {
      attrs.core_type = hw_thread.attrs.get_core_type();
      attrs.core_eff = hw_thread.attrs.get_core_eff();
      attrs.valid = 1;
    } else {
      // This mask spans across multiple attributes, set it as such
      if (attrs.core_type != hw_thread.attrs.get_core_type())
        attrs.core_type = KMP_HW_CORE_TYPE_UNKNOWN;
      if (attrs.core_eff != hw_thread.attrs.get_core_eff())
        attrs.core_eff = kmp_hw_attr_t::UNKNOWN_CORE_EFF;
    }
  }
}

static void __kmp_affinity_get_thread_topology_info(kmp_info_t *th) {
  if (!KMP_AFFINITY_CAPABLE())
    return;
  const kmp_affin_mask_t *mask = th->th.th_affin_mask;
  kmp_affinity_ids_t &ids = th->th.th_topology_ids;
  kmp_affinity_attrs_t &attrs = th->th.th_topology_attrs;
  __kmp_affinity_get_mask_topology_info(mask, ids, attrs);
}

// Assign the topology information to each place in the place list
// A thread can then grab not only its affinity mask, but the topology
// information associated with that mask. e.g., Which socket is a thread on
static void __kmp_affinity_get_topology_info(kmp_affinity_t &affinity) {
  if (!KMP_AFFINITY_CAPABLE())
    return;
  if (affinity.type != affinity_none) {
    KMP_ASSERT(affinity.num_os_id_masks);
    KMP_ASSERT(affinity.os_id_masks);
  }
  KMP_ASSERT(affinity.num_masks);
  KMP_ASSERT(affinity.masks);
  KMP_ASSERT(__kmp_affin_fullMask);

  int max_cpu = __kmp_affin_fullMask->get_max_cpu();
  int num_hw_threads = __kmp_topology->get_num_hw_threads();

  // Allocate thread topology information
  if (!affinity.ids) {
    affinity.ids = (kmp_affinity_ids_t *)__kmp_allocate(
        sizeof(kmp_affinity_ids_t) * affinity.num_masks);
  }
  if (!affinity.attrs) {
    affinity.attrs = (kmp_affinity_attrs_t *)__kmp_allocate(
        sizeof(kmp_affinity_attrs_t) * affinity.num_masks);
  }
  if (!__kmp_osid_to_hwthread_map) {
    // Want the +1 because max_cpu should be valid index into map
    __kmp_osid_to_hwthread_map =
        (int *)__kmp_allocate(sizeof(int) * (max_cpu + 1));
  }

  // Create the OS proc to hardware thread map
  for (int hw_thread = 0; hw_thread < num_hw_threads; ++hw_thread) {
    int os_id = __kmp_topology->at(hw_thread).os_id;
    if (KMP_CPU_ISSET(os_id, __kmp_affin_fullMask))
      __kmp_osid_to_hwthread_map[os_id] = hw_thread;
  }

  for (unsigned i = 0; i < affinity.num_masks; ++i) {
    kmp_affinity_ids_t &ids = affinity.ids[i];
    kmp_affinity_attrs_t &attrs = affinity.attrs[i];
    kmp_affin_mask_t *mask = KMP_CPU_INDEX(affinity.masks, i);
    __kmp_affinity_get_mask_topology_info(mask, ids, attrs);
  }
}

// Called when __kmp_topology is ready
static void __kmp_aux_affinity_initialize_other_data(kmp_affinity_t &affinity) {
  // Initialize other data structures which depend on the topology
  if (__kmp_topology && __kmp_topology->get_num_hw_threads()) {
    machine_hierarchy.init(__kmp_topology->get_num_hw_threads());
    __kmp_affinity_get_topology_info(affinity);
#if KMP_WEIGHTED_ITERATIONS_SUPPORTED
    __kmp_first_osid_with_ecore = __kmp_get_first_osid_with_ecore();
#endif
  }
}

// Create a one element mask array (set of places) which only contains the
// initial process's affinity mask
static void __kmp_create_affinity_none_places(kmp_affinity_t &affinity) {
  KMP_ASSERT(__kmp_affin_fullMask != NULL);
  KMP_ASSERT(affinity.type == affinity_none);
  KMP_ASSERT(__kmp_avail_proc == __kmp_topology->get_num_hw_threads());
  affinity.num_masks = 1;
  KMP_CPU_ALLOC_ARRAY(affinity.masks, affinity.num_masks);
  kmp_affin_mask_t *dest = KMP_CPU_INDEX(affinity.masks, 0);
  KMP_CPU_COPY(dest, __kmp_affin_fullMask);
  __kmp_aux_affinity_initialize_other_data(affinity);
}

static void __kmp_aux_affinity_initialize_masks(kmp_affinity_t &affinity) {
  // Create the "full" mask - this defines all of the processors that we
  // consider to be in the machine model. If respect is set, then it is the
  // initialization thread's affinity mask. Otherwise, it is all processors that
  // we know about on the machine.
  int verbose = affinity.flags.verbose;
  const char *env_var = affinity.env_var;

  // Already initialized
  if (__kmp_affin_fullMask && __kmp_affin_origMask)
    return;

  if (__kmp_affin_fullMask == NULL) {
    KMP_CPU_ALLOC(__kmp_affin_fullMask);
  }
  if (__kmp_affin_origMask == NULL) {
    KMP_CPU_ALLOC(__kmp_affin_origMask);
  }
  if (KMP_AFFINITY_CAPABLE()) {
    __kmp_get_system_affinity(__kmp_affin_fullMask, TRUE);
    // Make a copy before possible expanding to the entire machine mask
    __kmp_affin_origMask->copy(__kmp_affin_fullMask);
    if (affinity.flags.respect) {
      // Count the number of available processors.
      unsigned i;
      __kmp_avail_proc = 0;
      KMP_CPU_SET_ITERATE(i, __kmp_affin_fullMask) {
        if (!KMP_CPU_ISSET(i, __kmp_affin_fullMask)) {
          continue;
        }
        __kmp_avail_proc++;
      }
      if (__kmp_avail_proc > __kmp_xproc) {
        KMP_AFF_WARNING(affinity, ErrorInitializeAffinity);
        affinity.type = affinity_none;
        KMP_AFFINITY_DISABLE();
        return;
      }

      if (verbose) {
        char buf[KMP_AFFIN_MASK_PRINT_LEN];
        __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                                  __kmp_affin_fullMask);
        KMP_INFORM(InitOSProcSetRespect, env_var, buf);
      }
    } else {
      if (verbose) {
        char buf[KMP_AFFIN_MASK_PRINT_LEN];
        __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                                  __kmp_affin_fullMask);
        KMP_INFORM(InitOSProcSetNotRespect, env_var, buf);
      }
      __kmp_avail_proc =
          __kmp_affinity_entire_machine_mask(__kmp_affin_fullMask);
#if KMP_OS_WINDOWS
      if (__kmp_num_proc_groups <= 1) {
        // Copy expanded full mask if topology has single processor group
        __kmp_affin_origMask->copy(__kmp_affin_fullMask);
      }
      // Set the process affinity mask since threads' affinity
      // masks must be subset of process mask in Windows* OS
      __kmp_affin_fullMask->set_process_affinity(true);
#endif
    }
  }
}

static bool __kmp_aux_affinity_initialize_topology(kmp_affinity_t &affinity) {
  bool success = false;
  const char *env_var = affinity.env_var;
  kmp_i18n_id_t msg_id = kmp_i18n_null;
  int verbose = affinity.flags.verbose;

  // For backward compatibility, setting KMP_CPUINFO_FILE =>
  // KMP_TOPOLOGY_METHOD=cpuinfo
  if ((__kmp_cpuinfo_file != NULL) &&
      (__kmp_affinity_top_method == affinity_top_method_all)) {
    __kmp_affinity_top_method = affinity_top_method_cpuinfo;
  }

  if (__kmp_affinity_top_method == affinity_top_method_all) {
// In the default code path, errors are not fatal - we just try using
// another method. We only emit a warning message if affinity is on, or the
// verbose flag is set, an the nowarnings flag was not set.
#if KMP_USE_HWLOC
    if (!success &&
        __kmp_affinity_dispatch->get_api_type() == KMPAffinity::HWLOC) {
      if (!__kmp_hwloc_error) {
        success = __kmp_affinity_create_hwloc_map(&msg_id);
        if (!success && verbose) {
          KMP_INFORM(AffIgnoringHwloc, env_var);
        }
      } else if (verbose) {
        KMP_INFORM(AffIgnoringHwloc, env_var);
      }
    }
#endif

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
    if (!success) {
      success = __kmp_affinity_create_x2apicid_map(&msg_id);
      if (!success && verbose && msg_id != kmp_i18n_null) {
        KMP_INFORM(AffInfoStr, env_var, __kmp_i18n_catgets(msg_id));
      }
    }
    if (!success) {
      success = __kmp_affinity_create_apicid_map(&msg_id);
      if (!success && verbose && msg_id != kmp_i18n_null) {
        KMP_INFORM(AffInfoStr, env_var, __kmp_i18n_catgets(msg_id));
      }
    }
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

#if KMP_OS_LINUX || KMP_OS_AIX
    if (!success) {
      int line = 0;
      success = __kmp_affinity_create_cpuinfo_map(&line, &msg_id);
      if (!success && verbose && msg_id != kmp_i18n_null) {
        KMP_INFORM(AffInfoStr, env_var, __kmp_i18n_catgets(msg_id));
      }
    }
#endif /* KMP_OS_LINUX */

#if KMP_GROUP_AFFINITY
    if (!success && (__kmp_num_proc_groups > 1)) {
      success = __kmp_affinity_create_proc_group_map(&msg_id);
      if (!success && verbose && msg_id != kmp_i18n_null) {
        KMP_INFORM(AffInfoStr, env_var, __kmp_i18n_catgets(msg_id));
      }
    }
#endif /* KMP_GROUP_AFFINITY */

    if (!success) {
      success = __kmp_affinity_create_flat_map(&msg_id);
      if (!success && verbose && msg_id != kmp_i18n_null) {
        KMP_INFORM(AffInfoStr, env_var, __kmp_i18n_catgets(msg_id));
      }
      KMP_ASSERT(success);
    }
  }

// If the user has specified that a paricular topology discovery method is to be
// used, then we abort if that method fails. The exception is group affinity,
// which might have been implicitly set.
#if KMP_USE_HWLOC
  else if (__kmp_affinity_top_method == affinity_top_method_hwloc) {
    KMP_ASSERT(__kmp_affinity_dispatch->get_api_type() == KMPAffinity::HWLOC);
    success = __kmp_affinity_create_hwloc_map(&msg_id);
    if (!success) {
      KMP_ASSERT(msg_id != kmp_i18n_null);
      KMP_FATAL(MsgExiting, __kmp_i18n_catgets(msg_id));
    }
  }
#endif // KMP_USE_HWLOC

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
  else if (__kmp_affinity_top_method == affinity_top_method_x2apicid ||
           __kmp_affinity_top_method == affinity_top_method_x2apicid_1f) {
    success = __kmp_affinity_create_x2apicid_map(&msg_id);
    if (!success) {
      KMP_ASSERT(msg_id != kmp_i18n_null);
      KMP_FATAL(MsgExiting, __kmp_i18n_catgets(msg_id));
    }
  } else if (__kmp_affinity_top_method == affinity_top_method_apicid) {
    success = __kmp_affinity_create_apicid_map(&msg_id);
    if (!success) {
      KMP_ASSERT(msg_id != kmp_i18n_null);
      KMP_FATAL(MsgExiting, __kmp_i18n_catgets(msg_id));
    }
  }
#endif /* KMP_ARCH_X86 || KMP_ARCH_X86_64 */

  else if (__kmp_affinity_top_method == affinity_top_method_cpuinfo) {
    int line = 0;
    success = __kmp_affinity_create_cpuinfo_map(&line, &msg_id);
    if (!success) {
      KMP_ASSERT(msg_id != kmp_i18n_null);
      const char *filename = __kmp_cpuinfo_get_filename();
      if (line > 0) {
        KMP_FATAL(FileLineMsgExiting, filename, line,
                  __kmp_i18n_catgets(msg_id));
      } else {
        KMP_FATAL(FileMsgExiting, filename, __kmp_i18n_catgets(msg_id));
      }
    }
  }

#if KMP_GROUP_AFFINITY
  else if (__kmp_affinity_top_method == affinity_top_method_group) {
    success = __kmp_affinity_create_proc_group_map(&msg_id);
    KMP_ASSERT(success);
    if (!success) {
      KMP_ASSERT(msg_id != kmp_i18n_null);
      KMP_FATAL(MsgExiting, __kmp_i18n_catgets(msg_id));
    }
  }
#endif /* KMP_GROUP_AFFINITY */

  else if (__kmp_affinity_top_method == affinity_top_method_flat) {
    success = __kmp_affinity_create_flat_map(&msg_id);
    // should not fail
    KMP_ASSERT(success);
  }

  // Early exit if topology could not be created
  if (!__kmp_topology) {
    if (KMP_AFFINITY_CAPABLE()) {
      KMP_AFF_WARNING(affinity, ErrorInitializeAffinity);
    }
    if (nPackages > 0 && nCoresPerPkg > 0 && __kmp_nThreadsPerCore > 0 &&
        __kmp_ncores > 0) {
      __kmp_topology = kmp_topology_t::allocate(0, 0, NULL);
      __kmp_topology->canonicalize(nPackages, nCoresPerPkg,
                                   __kmp_nThreadsPerCore, __kmp_ncores);
      if (verbose) {
        __kmp_topology->print(env_var);
      }
    }
    return false;
  }

  // Canonicalize, print (if requested), apply KMP_HW_SUBSET
  __kmp_topology->canonicalize();
  if (verbose)
    __kmp_topology->print(env_var);
  bool filtered = __kmp_topology->filter_hw_subset();
  if (filtered && verbose)
    __kmp_topology->print("KMP_HW_SUBSET");
  return success;
}

static void __kmp_aux_affinity_initialize(kmp_affinity_t &affinity) {
  bool is_regular_affinity = (&affinity == &__kmp_affinity);
  bool is_hidden_helper_affinity = (&affinity == &__kmp_hh_affinity);
  const char *env_var = __kmp_get_affinity_env_var(affinity);

  if (affinity.flags.initialized) {
    KMP_ASSERT(__kmp_affin_fullMask != NULL);
    return;
  }

  if (is_regular_affinity && (!__kmp_affin_fullMask || !__kmp_affin_origMask))
    __kmp_aux_affinity_initialize_masks(affinity);

  if (is_regular_affinity && !__kmp_topology) {
    bool success = __kmp_aux_affinity_initialize_topology(affinity);
    if (success) {
      KMP_ASSERT(__kmp_avail_proc == __kmp_topology->get_num_hw_threads());
    } else {
      affinity.type = affinity_none;
      KMP_AFFINITY_DISABLE();
    }
  }

  // If KMP_AFFINITY=none, then only create the single "none" place
  // which is the process's initial affinity mask or the number of
  // hardware threads depending on respect,norespect
  if (affinity.type == affinity_none) {
    __kmp_create_affinity_none_places(affinity);
#if KMP_USE_HIER_SCHED
    __kmp_dispatch_set_hierarchy_values();
#endif
    affinity.flags.initialized = TRUE;
    return;
  }

  __kmp_topology->set_granularity(affinity);
  int depth = __kmp_topology->get_depth();

  // Create the table of masks, indexed by thread Id.
  unsigned numUnique;
  int numAddrs = __kmp_topology->get_num_hw_threads();
  // If OMP_PLACES=cores:<attribute> specified, then attempt
  // to make OS Id mask table using those attributes
  if (affinity.core_attr_gran.valid) {
    __kmp_create_os_id_masks(&numUnique, affinity, [&](int idx) {
      KMP_ASSERT(idx >= -1);
      for (int i = idx + 1; i < numAddrs; ++i)
        if (__kmp_topology->at(i).attrs.contains(affinity.core_attr_gran))
          return i;
      return numAddrs;
    });
    if (!affinity.os_id_masks) {
      const char *core_attribute;
      if (affinity.core_attr_gran.core_eff != kmp_hw_attr_t::UNKNOWN_CORE_EFF)
        core_attribute = "core_efficiency";
      else
        core_attribute = "core_type";
      KMP_AFF_WARNING(affinity, AffIgnoringNotAvailable, env_var,
                      core_attribute,
                      __kmp_hw_get_catalog_string(KMP_HW_CORE, /*plural=*/true))
    }
  }
  // If core attributes did not work, or none were specified,
  // then make OS Id mask table using typical incremental way.
  if (!affinity.os_id_masks) {
    __kmp_create_os_id_masks(&numUnique, affinity, [](int idx) {
      KMP_ASSERT(idx >= -1);
      return idx + 1;
    });
  }
  if (affinity.gran_levels == 0) {
    KMP_DEBUG_ASSERT((int)numUnique == __kmp_avail_proc);
  }

  switch (affinity.type) {

  case affinity_explicit:
    KMP_DEBUG_ASSERT(affinity.proclist != NULL);
    if (is_hidden_helper_affinity ||
        __kmp_nested_proc_bind.bind_types[0] == proc_bind_intel) {
      __kmp_affinity_process_proclist(affinity);
    } else {
      __kmp_affinity_process_placelist(affinity);
    }
    if (affinity.num_masks == 0) {
      KMP_AFF_WARNING(affinity, AffNoValidProcID);
      affinity.type = affinity_none;
      __kmp_create_affinity_none_places(affinity);
      affinity.flags.initialized = TRUE;
      return;
    }
    break;

  // The other affinity types rely on sorting the hardware threads according to
  // some permutation of the machine topology tree. Set affinity.compact
  // and affinity.offset appropriately, then jump to a common code
  // fragment to do the sort and create the array of affinity masks.
  case affinity_logical:
    affinity.compact = 0;
    if (affinity.offset) {
      affinity.offset =
          __kmp_nThreadsPerCore * affinity.offset % __kmp_avail_proc;
    }
    goto sortTopology;

  case affinity_physical:
    if (__kmp_nThreadsPerCore > 1) {
      affinity.compact = 1;
      if (affinity.compact >= depth) {
        affinity.compact = 0;
      }
    } else {
      affinity.compact = 0;
    }
    if (affinity.offset) {
      affinity.offset =
          __kmp_nThreadsPerCore * affinity.offset % __kmp_avail_proc;
    }
    goto sortTopology;

  case affinity_scatter:
    if (affinity.compact >= depth) {
      affinity.compact = 0;
    } else {
      affinity.compact = depth - 1 - affinity.compact;
    }
    goto sortTopology;

  case affinity_compact:
    if (affinity.compact >= depth) {
      affinity.compact = depth - 1;
    }
    goto sortTopology;

  case affinity_balanced:
    if (depth <= 1 || is_hidden_helper_affinity) {
      KMP_AFF_WARNING(affinity, AffBalancedNotAvail, env_var);
      affinity.type = affinity_none;
      __kmp_create_affinity_none_places(affinity);
      affinity.flags.initialized = TRUE;
      return;
    } else if (!__kmp_topology->is_uniform()) {
      // Save the depth for further usage
      __kmp_aff_depth = depth;

      int core_level =
          __kmp_affinity_find_core_level(__kmp_avail_proc, depth - 1);
      int ncores = __kmp_affinity_compute_ncores(__kmp_avail_proc, depth - 1,
                                                 core_level);
      int maxprocpercore = __kmp_affinity_max_proc_per_core(
          __kmp_avail_proc, depth - 1, core_level);

      int nproc = ncores * maxprocpercore;
      if ((nproc < 2) || (nproc < __kmp_avail_proc)) {
        KMP_AFF_WARNING(affinity, AffBalancedNotAvail, env_var);
        affinity.type = affinity_none;
        __kmp_create_affinity_none_places(affinity);
        affinity.flags.initialized = TRUE;
        return;
      }

      procarr = (int *)__kmp_allocate(sizeof(int) * nproc);
      for (int i = 0; i < nproc; i++) {
        procarr[i] = -1;
      }

      int lastcore = -1;
      int inlastcore = 0;
      for (int i = 0; i < __kmp_avail_proc; i++) {
        int proc = __kmp_topology->at(i).os_id;
        int core = __kmp_affinity_find_core(i, depth - 1, core_level);

        if (core == lastcore) {
          inlastcore++;
        } else {
          inlastcore = 0;
        }
        lastcore = core;

        procarr[core * maxprocpercore + inlastcore] = proc;
      }
    }
    if (affinity.compact >= depth) {
      affinity.compact = depth - 1;
    }

  sortTopology:
    // Allocate the gtid->affinity mask table.
    if (affinity.flags.dups) {
      affinity.num_masks = __kmp_avail_proc;
    } else {
      affinity.num_masks = numUnique;
    }

    if ((__kmp_nested_proc_bind.bind_types[0] != proc_bind_intel) &&
        (__kmp_affinity_num_places > 0) &&
        ((unsigned)__kmp_affinity_num_places < affinity.num_masks) &&
        !is_hidden_helper_affinity) {
      affinity.num_masks = __kmp_affinity_num_places;
    }

    KMP_CPU_ALLOC_ARRAY(affinity.masks, affinity.num_masks);

    // Sort the topology table according to the current setting of
    // affinity.compact, then fill out affinity.masks.
    __kmp_topology->sort_compact(affinity);
    {
      int i;
      unsigned j;
      int num_hw_threads = __kmp_topology->get_num_hw_threads();
      kmp_full_mask_modifier_t full_mask;
      for (i = 0, j = 0; i < num_hw_threads; i++) {
        if ((!affinity.flags.dups) && (!__kmp_topology->at(i).leader)) {
          continue;
        }
        int osId = __kmp_topology->at(i).os_id;

        kmp_affin_mask_t *src = KMP_CPU_INDEX(affinity.os_id_masks, osId);
        kmp_affin_mask_t *dest = KMP_CPU_INDEX(affinity.masks, j);
        KMP_ASSERT(KMP_CPU_ISSET(osId, src));
        KMP_CPU_COPY(dest, src);
        full_mask.include(src);
        if (++j >= affinity.num_masks) {
          break;
        }
      }
      KMP_DEBUG_ASSERT(j == affinity.num_masks);
      // See if the places list further restricts or changes the full mask
      if (full_mask.restrict_to_mask() && affinity.flags.verbose) {
        __kmp_topology->print(env_var);
      }
    }
    // Sort the topology back using ids
    __kmp_topology->sort_ids();
    break;

  default:
    KMP_ASSERT2(0, "Unexpected affinity setting");
  }
  __kmp_aux_affinity_initialize_other_data(affinity);
  affinity.flags.initialized = TRUE;
}

void __kmp_affinity_initialize(kmp_affinity_t &affinity) {
  // Much of the code above was written assuming that if a machine was not
  // affinity capable, then affinity type == affinity_none.
  // We now explicitly represent this as affinity type == affinity_disabled.
  // There are too many checks for affinity type == affinity_none in this code.
  // Instead of trying to change them all, check if
  // affinity type == affinity_disabled, and if so, slam it with affinity_none,
  // call the real initialization routine, then restore affinity type to
  // affinity_disabled.
  int disabled = (affinity.type == affinity_disabled);
  if (!KMP_AFFINITY_CAPABLE())
    KMP_ASSERT(disabled);
  if (disabled)
    affinity.type = affinity_none;
  __kmp_aux_affinity_initialize(affinity);
  if (disabled)
    affinity.type = affinity_disabled;
}

void __kmp_affinity_uninitialize(void) {
  for (kmp_affinity_t *affinity : __kmp_affinities) {
    if (affinity->masks != NULL)
      KMP_CPU_FREE_ARRAY(affinity->masks, affinity->num_masks);
    if (affinity->os_id_masks != NULL)
      KMP_CPU_FREE_ARRAY(affinity->os_id_masks, affinity->num_os_id_masks);
    if (affinity->proclist != NULL)
      __kmp_free(affinity->proclist);
    if (affinity->ids != NULL)
      __kmp_free(affinity->ids);
    if (affinity->attrs != NULL)
      __kmp_free(affinity->attrs);
    *affinity = KMP_AFFINITY_INIT(affinity->env_var);
  }
  if (__kmp_affin_origMask != NULL) {
    if (KMP_AFFINITY_CAPABLE()) {
#if KMP_OS_AIX
      // Uninitialize by unbinding the thread.
      bindprocessor(BINDTHREAD, thread_self(), PROCESSOR_CLASS_ANY);
#else
      __kmp_set_system_affinity(__kmp_affin_origMask, FALSE);
#endif
    }
    KMP_CPU_FREE(__kmp_affin_origMask);
    __kmp_affin_origMask = NULL;
  }
  __kmp_affinity_num_places = 0;
  if (procarr != NULL) {
    __kmp_free(procarr);
    procarr = NULL;
  }
  if (__kmp_osid_to_hwthread_map) {
    __kmp_free(__kmp_osid_to_hwthread_map);
    __kmp_osid_to_hwthread_map = NULL;
  }
#if KMP_USE_HWLOC
  if (__kmp_hwloc_topology != NULL) {
    hwloc_topology_destroy(__kmp_hwloc_topology);
    __kmp_hwloc_topology = NULL;
  }
#endif
  if (__kmp_hw_subset) {
    kmp_hw_subset_t::deallocate(__kmp_hw_subset);
    __kmp_hw_subset = nullptr;
  }
  if (__kmp_topology) {
    kmp_topology_t::deallocate(__kmp_topology);
    __kmp_topology = nullptr;
  }
  KMPAffinity::destroy_api();
}

static void __kmp_select_mask_by_gtid(int gtid, const kmp_affinity_t *affinity,
                                      int *place, kmp_affin_mask_t **mask) {
  int mask_idx;
  bool is_hidden_helper = KMP_HIDDEN_HELPER_THREAD(gtid);
  if (is_hidden_helper)
    // The first gtid is the regular primary thread, the second gtid is the main
    // thread of hidden team which does not participate in task execution.
    mask_idx = gtid - 2;
  else
    mask_idx = __kmp_adjust_gtid_for_hidden_helpers(gtid);
  KMP_DEBUG_ASSERT(affinity->num_masks > 0);
  *place = (mask_idx + affinity->offset) % affinity->num_masks;
  *mask = KMP_CPU_INDEX(affinity->masks, *place);
}

// This function initializes the per-thread data concerning affinity including
// the mask and topology information
void __kmp_affinity_set_init_mask(int gtid, int isa_root) {

  kmp_info_t *th = (kmp_info_t *)TCR_SYNC_PTR(__kmp_threads[gtid]);

  // Set the thread topology information to default of unknown
  for (int id = 0; id < KMP_HW_LAST; ++id)
    th->th.th_topology_ids.ids[id] = kmp_hw_thread_t::UNKNOWN_ID;
  th->th.th_topology_attrs = KMP_AFFINITY_ATTRS_UNKNOWN;

  if (!KMP_AFFINITY_CAPABLE()) {
    return;
  }

  if (th->th.th_affin_mask == NULL) {
    KMP_CPU_ALLOC(th->th.th_affin_mask);
  } else {
    KMP_CPU_ZERO(th->th.th_affin_mask);
  }

  // Copy the thread mask to the kmp_info_t structure. If
  // __kmp_affinity.type == affinity_none, copy the "full" mask, i.e.
  // one that has all of the OS proc ids set, or if
  // __kmp_affinity.flags.respect is set, then the full mask is the
  // same as the mask of the initialization thread.
  kmp_affin_mask_t *mask;
  int i;
  const kmp_affinity_t *affinity;
  bool is_hidden_helper = KMP_HIDDEN_HELPER_THREAD(gtid);

  if (is_hidden_helper)
    affinity = &__kmp_hh_affinity;
  else
    affinity = &__kmp_affinity;

  if (KMP_AFFINITY_NON_PROC_BIND || is_hidden_helper) {
    if ((affinity->type == affinity_none) ||
        (affinity->type == affinity_balanced) ||
        KMP_HIDDEN_HELPER_MAIN_THREAD(gtid)) {
#if KMP_GROUP_AFFINITY
      if (__kmp_num_proc_groups > 1) {
        return;
      }
#endif
      KMP_ASSERT(__kmp_affin_fullMask != NULL);
      i = 0;
      mask = __kmp_affin_fullMask;
    } else {
      __kmp_select_mask_by_gtid(gtid, affinity, &i, &mask);
    }
  } else {
    if (!isa_root || __kmp_nested_proc_bind.bind_types[0] == proc_bind_false) {
#if KMP_GROUP_AFFINITY
      if (__kmp_num_proc_groups > 1) {
        return;
      }
#endif
      KMP_ASSERT(__kmp_affin_fullMask != NULL);
      i = KMP_PLACE_ALL;
      mask = __kmp_affin_fullMask;
    } else {
      __kmp_select_mask_by_gtid(gtid, affinity, &i, &mask);
    }
  }

  th->th.th_current_place = i;
  if (isa_root && !is_hidden_helper) {
    th->th.th_new_place = i;
    th->th.th_first_place = 0;
    th->th.th_last_place = affinity->num_masks - 1;
  } else if (KMP_AFFINITY_NON_PROC_BIND) {
    // When using a Non-OMP_PROC_BIND affinity method,
    // set all threads' place-partition-var to the entire place list
    th->th.th_first_place = 0;
    th->th.th_last_place = affinity->num_masks - 1;
  }
  // Copy topology information associated with the place
  if (i >= 0) {
    th->th.th_topology_ids = __kmp_affinity.ids[i];
    th->th.th_topology_attrs = __kmp_affinity.attrs[i];
  }

  if (i == KMP_PLACE_ALL) {
    KA_TRACE(100, ("__kmp_affinity_set_init_mask: setting T#%d to all places\n",
                   gtid));
  } else {
    KA_TRACE(100, ("__kmp_affinity_set_init_mask: setting T#%d to place %d\n",
                   gtid, i));
  }

  KMP_CPU_COPY(th->th.th_affin_mask, mask);
}

void __kmp_affinity_bind_init_mask(int gtid) {
  if (!KMP_AFFINITY_CAPABLE()) {
    return;
  }
  kmp_info_t *th = (kmp_info_t *)TCR_SYNC_PTR(__kmp_threads[gtid]);
  const kmp_affinity_t *affinity;
  const char *env_var;
  bool is_hidden_helper = KMP_HIDDEN_HELPER_THREAD(gtid);

  if (is_hidden_helper)
    affinity = &__kmp_hh_affinity;
  else
    affinity = &__kmp_affinity;
  env_var = __kmp_get_affinity_env_var(*affinity, /*for_binding=*/true);
  /* to avoid duplicate printing (will be correctly printed on barrier) */
  if (affinity->flags.verbose && (affinity->type == affinity_none ||
                                  (th->th.th_current_place != KMP_PLACE_ALL &&
                                   affinity->type != affinity_balanced)) &&
      !KMP_HIDDEN_HELPER_MAIN_THREAD(gtid)) {
    char buf[KMP_AFFIN_MASK_PRINT_LEN];
    __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                              th->th.th_affin_mask);
    KMP_INFORM(BoundToOSProcSet, env_var, (kmp_int32)getpid(), __kmp_gettid(),
               gtid, buf);
  }

#if KMP_OS_WINDOWS
  // On Windows* OS, the process affinity mask might have changed. If the user
  // didn't request affinity and this call fails, just continue silently.
  // See CQ171393.
  if (affinity->type == affinity_none) {
    __kmp_set_system_affinity(th->th.th_affin_mask, FALSE);
  } else
#endif
#ifndef KMP_OS_AIX
    // Do not set the full mask as the init mask on AIX.
    __kmp_set_system_affinity(th->th.th_affin_mask, TRUE);
#endif
}

void __kmp_affinity_bind_place(int gtid) {
  // Hidden helper threads should not be affected by OMP_PLACES/OMP_PROC_BIND
  if (!KMP_AFFINITY_CAPABLE() || KMP_HIDDEN_HELPER_THREAD(gtid)) {
    return;
  }

  kmp_info_t *th = (kmp_info_t *)TCR_SYNC_PTR(__kmp_threads[gtid]);

  KA_TRACE(100, ("__kmp_affinity_bind_place: binding T#%d to place %d (current "
                 "place = %d)\n",
                 gtid, th->th.th_new_place, th->th.th_current_place));

  // Check that the new place is within this thread's partition.
  KMP_DEBUG_ASSERT(th->th.th_affin_mask != NULL);
  KMP_ASSERT(th->th.th_new_place >= 0);
  KMP_ASSERT((unsigned)th->th.th_new_place <= __kmp_affinity.num_masks);
  if (th->th.th_first_place <= th->th.th_last_place) {
    KMP_ASSERT((th->th.th_new_place >= th->th.th_first_place) &&
               (th->th.th_new_place <= th->th.th_last_place));
  } else {
    KMP_ASSERT((th->th.th_new_place <= th->th.th_first_place) ||
               (th->th.th_new_place >= th->th.th_last_place));
  }

  // Copy the thread mask to the kmp_info_t structure,
  // and set this thread's affinity.
  kmp_affin_mask_t *mask =
      KMP_CPU_INDEX(__kmp_affinity.masks, th->th.th_new_place);
  KMP_CPU_COPY(th->th.th_affin_mask, mask);
  th->th.th_current_place = th->th.th_new_place;

  if (__kmp_affinity.flags.verbose) {
    char buf[KMP_AFFIN_MASK_PRINT_LEN];
    __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                              th->th.th_affin_mask);
    KMP_INFORM(BoundToOSProcSet, "OMP_PROC_BIND", (kmp_int32)getpid(),
               __kmp_gettid(), gtid, buf);
  }
  __kmp_set_system_affinity(th->th.th_affin_mask, TRUE);
}

int __kmp_aux_set_affinity(void **mask) {
  int gtid;
  kmp_info_t *th;
  int retval;

  if (!KMP_AFFINITY_CAPABLE()) {
    return -1;
  }

  gtid = __kmp_entry_gtid();
  KA_TRACE(
      1000, (""); {
        char buf[KMP_AFFIN_MASK_PRINT_LEN];
        __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                                  (kmp_affin_mask_t *)(*mask));
        __kmp_debug_printf(
            "kmp_set_affinity: setting affinity mask for thread %d = %s\n",
            gtid, buf);
      });

  if (__kmp_env_consistency_check) {
    if ((mask == NULL) || (*mask == NULL)) {
      KMP_FATAL(AffinityInvalidMask, "kmp_set_affinity");
    } else {
      unsigned proc;
      int num_procs = 0;

      KMP_CPU_SET_ITERATE(proc, ((kmp_affin_mask_t *)(*mask))) {
        if (!KMP_CPU_ISSET(proc, __kmp_affin_fullMask)) {
          KMP_FATAL(AffinityInvalidMask, "kmp_set_affinity");
        }
        if (!KMP_CPU_ISSET(proc, (kmp_affin_mask_t *)(*mask))) {
          continue;
        }
        num_procs++;
      }
      if (num_procs == 0) {
        KMP_FATAL(AffinityInvalidMask, "kmp_set_affinity");
      }

#if KMP_GROUP_AFFINITY
      if (__kmp_get_proc_group((kmp_affin_mask_t *)(*mask)) < 0) {
        KMP_FATAL(AffinityInvalidMask, "kmp_set_affinity");
      }
#endif /* KMP_GROUP_AFFINITY */
    }
  }

  th = __kmp_threads[gtid];
  KMP_DEBUG_ASSERT(th->th.th_affin_mask != NULL);
  retval = __kmp_set_system_affinity((kmp_affin_mask_t *)(*mask), FALSE);
  if (retval == 0) {
    KMP_CPU_COPY(th->th.th_affin_mask, (kmp_affin_mask_t *)(*mask));
  }

  th->th.th_current_place = KMP_PLACE_UNDEFINED;
  th->th.th_new_place = KMP_PLACE_UNDEFINED;
  th->th.th_first_place = 0;
  th->th.th_last_place = __kmp_affinity.num_masks - 1;

  // Turn off 4.0 affinity for the current tread at this parallel level.
  th->th.th_current_task->td_icvs.proc_bind = proc_bind_false;

  return retval;
}

int __kmp_aux_get_affinity(void **mask) {
  int gtid;
  int retval;
#if KMP_OS_WINDOWS || KMP_OS_AIX || KMP_DEBUG
  kmp_info_t *th;
#endif
  if (!KMP_AFFINITY_CAPABLE()) {
    return -1;
  }

  gtid = __kmp_entry_gtid();
#if KMP_OS_WINDOWS || KMP_OS_AIX || KMP_DEBUG
  th = __kmp_threads[gtid];
#else
  (void)gtid; // unused variable
#endif
  KMP_DEBUG_ASSERT(th->th.th_affin_mask != NULL);

  KA_TRACE(
      1000, (""); {
        char buf[KMP_AFFIN_MASK_PRINT_LEN];
        __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                                  th->th.th_affin_mask);
        __kmp_printf(
            "kmp_get_affinity: stored affinity mask for thread %d = %s\n", gtid,
            buf);
      });

  if (__kmp_env_consistency_check) {
    if ((mask == NULL) || (*mask == NULL)) {
      KMP_FATAL(AffinityInvalidMask, "kmp_get_affinity");
    }
  }

#if !KMP_OS_WINDOWS && !KMP_OS_AIX

  retval = __kmp_get_system_affinity((kmp_affin_mask_t *)(*mask), FALSE);
  KA_TRACE(
      1000, (""); {
        char buf[KMP_AFFIN_MASK_PRINT_LEN];
        __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                                  (kmp_affin_mask_t *)(*mask));
        __kmp_printf(
            "kmp_get_affinity: system affinity mask for thread %d = %s\n", gtid,
            buf);
      });
  return retval;

#else
  (void)retval;

  KMP_CPU_COPY((kmp_affin_mask_t *)(*mask), th->th.th_affin_mask);
  return 0;

#endif /* !KMP_OS_WINDOWS && !KMP_OS_AIX */
}

int __kmp_aux_get_affinity_max_proc() {
  if (!KMP_AFFINITY_CAPABLE()) {
    return 0;
  }
#if KMP_GROUP_AFFINITY
  if (__kmp_num_proc_groups > 1) {
    return (int)(__kmp_num_proc_groups * sizeof(DWORD_PTR) * CHAR_BIT);
  }
#endif
  return __kmp_xproc;
}

int __kmp_aux_set_affinity_mask_proc(int proc, void **mask) {
  if (!KMP_AFFINITY_CAPABLE()) {
    return -1;
  }

  KA_TRACE(
      1000, (""); {
        int gtid = __kmp_entry_gtid();
        char buf[KMP_AFFIN_MASK_PRINT_LEN];
        __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                                  (kmp_affin_mask_t *)(*mask));
        __kmp_debug_printf("kmp_set_affinity_mask_proc: setting proc %d in "
                           "affinity mask for thread %d = %s\n",
                           proc, gtid, buf);
      });

  if (__kmp_env_consistency_check) {
    if ((mask == NULL) || (*mask == NULL)) {
      KMP_FATAL(AffinityInvalidMask, "kmp_set_affinity_mask_proc");
    }
  }

  if ((proc < 0) || (proc >= __kmp_aux_get_affinity_max_proc())) {
    return -1;
  }
  if (!KMP_CPU_ISSET(proc, __kmp_affin_fullMask)) {
    return -2;
  }

  KMP_CPU_SET(proc, (kmp_affin_mask_t *)(*mask));
  return 0;
}

int __kmp_aux_unset_affinity_mask_proc(int proc, void **mask) {
  if (!KMP_AFFINITY_CAPABLE()) {
    return -1;
  }

  KA_TRACE(
      1000, (""); {
        int gtid = __kmp_entry_gtid();
        char buf[KMP_AFFIN_MASK_PRINT_LEN];
        __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                                  (kmp_affin_mask_t *)(*mask));
        __kmp_debug_printf("kmp_unset_affinity_mask_proc: unsetting proc %d in "
                           "affinity mask for thread %d = %s\n",
                           proc, gtid, buf);
      });

  if (__kmp_env_consistency_check) {
    if ((mask == NULL) || (*mask == NULL)) {
      KMP_FATAL(AffinityInvalidMask, "kmp_unset_affinity_mask_proc");
    }
  }

  if ((proc < 0) || (proc >= __kmp_aux_get_affinity_max_proc())) {
    return -1;
  }
  if (!KMP_CPU_ISSET(proc, __kmp_affin_fullMask)) {
    return -2;
  }

  KMP_CPU_CLR(proc, (kmp_affin_mask_t *)(*mask));
  return 0;
}

int __kmp_aux_get_affinity_mask_proc(int proc, void **mask) {
  if (!KMP_AFFINITY_CAPABLE()) {
    return -1;
  }

  KA_TRACE(
      1000, (""); {
        int gtid = __kmp_entry_gtid();
        char buf[KMP_AFFIN_MASK_PRINT_LEN];
        __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN,
                                  (kmp_affin_mask_t *)(*mask));
        __kmp_debug_printf("kmp_get_affinity_mask_proc: getting proc %d in "
                           "affinity mask for thread %d = %s\n",
                           proc, gtid, buf);
      });

  if (__kmp_env_consistency_check) {
    if ((mask == NULL) || (*mask == NULL)) {
      KMP_FATAL(AffinityInvalidMask, "kmp_get_affinity_mask_proc");
    }
  }

  if ((proc < 0) || (proc >= __kmp_aux_get_affinity_max_proc())) {
    return -1;
  }
  if (!KMP_CPU_ISSET(proc, __kmp_affin_fullMask)) {
    return 0;
  }

  return KMP_CPU_ISSET(proc, (kmp_affin_mask_t *)(*mask));
}

#if KMP_WEIGHTED_ITERATIONS_SUPPORTED
// Returns first os proc id with ATOM core
int __kmp_get_first_osid_with_ecore(void) {
  int low = 0;
  int high = __kmp_topology->get_num_hw_threads() - 1;
  int mid = 0;
  while (high - low > 1) {
    mid = (high + low) / 2;
    if (__kmp_topology->at(mid).attrs.get_core_type() ==
        KMP_HW_CORE_TYPE_CORE) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  if (__kmp_topology->at(mid).attrs.get_core_type() == KMP_HW_CORE_TYPE_ATOM) {
    return mid;
  }
  return -1;
}
#endif

// Dynamic affinity settings - Affinity balanced
void __kmp_balanced_affinity(kmp_info_t *th, int nthreads) {
  KMP_DEBUG_ASSERT(th);
  bool fine_gran = true;
  int tid = th->th.th_info.ds.ds_tid;
  const char *env_var = "KMP_AFFINITY";

  // Do not perform balanced affinity for the hidden helper threads
  if (KMP_HIDDEN_HELPER_THREAD(__kmp_gtid_from_thread(th)))
    return;

  switch (__kmp_affinity.gran) {
  case KMP_HW_THREAD:
    break;
  case KMP_HW_CORE:
    if (__kmp_nThreadsPerCore > 1) {
      fine_gran = false;
    }
    break;
  case KMP_HW_SOCKET:
    if (nCoresPerPkg > 1) {
      fine_gran = false;
    }
    break;
  default:
    fine_gran = false;
  }

  if (__kmp_topology->is_uniform()) {
    int coreID;
    int threadID;
    // Number of hyper threads per core in HT machine
    int __kmp_nth_per_core = __kmp_avail_proc / __kmp_ncores;
    // Number of cores
    int ncores = __kmp_ncores;
    if ((nPackages > 1) && (__kmp_nth_per_core <= 1)) {
      __kmp_nth_per_core = __kmp_avail_proc / nPackages;
      ncores = nPackages;
    }
    // How many threads will be bound to each core
    int chunk = nthreads / ncores;
    // How many cores will have an additional thread bound to it - "big cores"
    int big_cores = nthreads % ncores;
    // Number of threads on the big cores
    int big_nth = (chunk + 1) * big_cores;
    if (tid < big_nth) {
      coreID = tid / (chunk + 1);
      threadID = (tid % (chunk + 1)) % __kmp_nth_per_core;
    } else { // tid >= big_nth
      coreID = (tid - big_cores) / chunk;
      threadID = ((tid - big_cores) % chunk) % __kmp_nth_per_core;
    }
    KMP_DEBUG_ASSERT2(KMP_AFFINITY_CAPABLE(),
                      "Illegal set affinity operation when not capable");

    kmp_affin_mask_t *mask = th->th.th_affin_mask;
    KMP_CPU_ZERO(mask);

    if (fine_gran) {
      int osID =
          __kmp_topology->at(coreID * __kmp_nth_per_core + threadID).os_id;
      KMP_CPU_SET(osID, mask);
    } else {
      for (int i = 0; i < __kmp_nth_per_core; i++) {
        int osID;
        osID = __kmp_topology->at(coreID * __kmp_nth_per_core + i).os_id;
        KMP_CPU_SET(osID, mask);
      }
    }
    if (__kmp_affinity.flags.verbose) {
      char buf[KMP_AFFIN_MASK_PRINT_LEN];
      __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN, mask);
      KMP_INFORM(BoundToOSProcSet, env_var, (kmp_int32)getpid(), __kmp_gettid(),
                 tid, buf);
    }
    __kmp_affinity_get_thread_topology_info(th);
    __kmp_set_system_affinity(mask, TRUE);
  } else { // Non-uniform topology

    kmp_affin_mask_t *mask = th->th.th_affin_mask;
    KMP_CPU_ZERO(mask);

    int core_level =
        __kmp_affinity_find_core_level(__kmp_avail_proc, __kmp_aff_depth - 1);
    int ncores = __kmp_affinity_compute_ncores(__kmp_avail_proc,
                                               __kmp_aff_depth - 1, core_level);
    int nth_per_core = __kmp_affinity_max_proc_per_core(
        __kmp_avail_proc, __kmp_aff_depth - 1, core_level);

    // For performance gain consider the special case nthreads ==
    // __kmp_avail_proc
    if (nthreads == __kmp_avail_proc) {
      if (fine_gran) {
        int osID = __kmp_topology->at(tid).os_id;
        KMP_CPU_SET(osID, mask);
      } else {
        int core =
            __kmp_affinity_find_core(tid, __kmp_aff_depth - 1, core_level);
        for (int i = 0; i < __kmp_avail_proc; i++) {
          int osID = __kmp_topology->at(i).os_id;
          if (__kmp_affinity_find_core(i, __kmp_aff_depth - 1, core_level) ==
              core) {
            KMP_CPU_SET(osID, mask);
          }
        }
      }
    } else if (nthreads <= ncores) {

      int core = 0;
      for (int i = 0; i < ncores; i++) {
        // Check if this core from procarr[] is in the mask
        int in_mask = 0;
        for (int j = 0; j < nth_per_core; j++) {
          if (procarr[i * nth_per_core + j] != -1) {
            in_mask = 1;
            break;
          }
        }
        if (in_mask) {
          if (tid == core) {
            for (int j = 0; j < nth_per_core; j++) {
              int osID = procarr[i * nth_per_core + j];
              if (osID != -1) {
                KMP_CPU_SET(osID, mask);
                // For fine granularity it is enough to set the first available
                // osID for this core
                if (fine_gran) {
                  break;
                }
              }
            }
            break;
          } else {
            core++;
          }
        }
      }
    } else { // nthreads > ncores
      // Array to save the number of processors at each core
      int *nproc_at_core = (int *)KMP_ALLOCA(sizeof(int) * ncores);
      // Array to save the number of cores with "x" available processors;
      int *ncores_with_x_procs =
          (int *)KMP_ALLOCA(sizeof(int) * (nth_per_core + 1));
      // Array to save the number of cores with # procs from x to nth_per_core
      int *ncores_with_x_to_max_procs =
          (int *)KMP_ALLOCA(sizeof(int) * (nth_per_core + 1));

      for (int i = 0; i <= nth_per_core; i++) {
        ncores_with_x_procs[i] = 0;
        ncores_with_x_to_max_procs[i] = 0;
      }

      for (int i = 0; i < ncores; i++) {
        int cnt = 0;
        for (int j = 0; j < nth_per_core; j++) {
          if (procarr[i * nth_per_core + j] != -1) {
            cnt++;
          }
        }
        nproc_at_core[i] = cnt;
        ncores_with_x_procs[cnt]++;
      }

      for (int i = 0; i <= nth_per_core; i++) {
        for (int j = i; j <= nth_per_core; j++) {
          ncores_with_x_to_max_procs[i] += ncores_with_x_procs[j];
        }
      }

      // Max number of processors
      int nproc = nth_per_core * ncores;
      // An array to keep number of threads per each context
      int *newarr = (int *)__kmp_allocate(sizeof(int) * nproc);
      for (int i = 0; i < nproc; i++) {
        newarr[i] = 0;
      }

      int nth = nthreads;
      int flag = 0;
      while (nth > 0) {
        for (int j = 1; j <= nth_per_core; j++) {
          int cnt = ncores_with_x_to_max_procs[j];
          for (int i = 0; i < ncores; i++) {
            // Skip the core with 0 processors
            if (nproc_at_core[i] == 0) {
              continue;
            }
            for (int k = 0; k < nth_per_core; k++) {
              if (procarr[i * nth_per_core + k] != -1) {
                if (newarr[i * nth_per_core + k] == 0) {
                  newarr[i * nth_per_core + k] = 1;
                  cnt--;
                  nth--;
                  break;
                } else {
                  if (flag != 0) {
                    newarr[i * nth_per_core + k]++;
                    cnt--;
                    nth--;
                    break;
                  }
                }
              }
            }
            if (cnt == 0 || nth == 0) {
              break;
            }
          }
          if (nth == 0) {
            break;
          }
        }
        flag = 1;
      }
      int sum = 0;
      for (int i = 0; i < nproc; i++) {
        sum += newarr[i];
        if (sum > tid) {
          if (fine_gran) {
            int osID = procarr[i];
            KMP_CPU_SET(osID, mask);
          } else {
            int coreID = i / nth_per_core;
            for (int ii = 0; ii < nth_per_core; ii++) {
              int osID = procarr[coreID * nth_per_core + ii];
              if (osID != -1) {
                KMP_CPU_SET(osID, mask);
              }
            }
          }
          break;
        }
      }
      __kmp_free(newarr);
    }

    if (__kmp_affinity.flags.verbose) {
      char buf[KMP_AFFIN_MASK_PRINT_LEN];
      __kmp_affinity_print_mask(buf, KMP_AFFIN_MASK_PRINT_LEN, mask);
      KMP_INFORM(BoundToOSProcSet, env_var, (kmp_int32)getpid(), __kmp_gettid(),
                 tid, buf);
    }
    __kmp_affinity_get_thread_topology_info(th);
    __kmp_set_system_affinity(mask, TRUE);
  }
}

#if KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_AIX
// We don't need this entry for Windows because
// there is GetProcessAffinityMask() api
//
// The intended usage is indicated by these steps:
// 1) The user gets the current affinity mask
// 2) Then sets the affinity by calling this function
// 3) Error check the return value
// 4) Use non-OpenMP parallelization
// 5) Reset the affinity to what was stored in step 1)
#ifdef __cplusplus
extern "C"
#endif
    int
    kmp_set_thread_affinity_mask_initial()
// the function returns 0 on success,
//   -1 if we cannot bind thread
//   >0 (errno) if an error happened during binding
{
  int gtid = __kmp_get_gtid();
  if (gtid < 0) {
    // Do not touch non-omp threads
    KA_TRACE(30, ("kmp_set_thread_affinity_mask_initial: "
                  "non-omp thread, returning\n"));
    return -1;
  }
  if (!KMP_AFFINITY_CAPABLE() || !__kmp_init_middle) {
    KA_TRACE(30, ("kmp_set_thread_affinity_mask_initial: "
                  "affinity not initialized, returning\n"));
    return -1;
  }
  KA_TRACE(30, ("kmp_set_thread_affinity_mask_initial: "
                "set full mask for thread %d\n",
                gtid));
  KMP_DEBUG_ASSERT(__kmp_affin_fullMask != NULL);
#if KMP_OS_AIX
  return bindprocessor(BINDTHREAD, thread_self(), PROCESSOR_CLASS_ANY);
#else
  return __kmp_set_system_affinity(__kmp_affin_fullMask, FALSE);
#endif
}
#endif

#endif // KMP_AFFINITY_SUPPORTED
