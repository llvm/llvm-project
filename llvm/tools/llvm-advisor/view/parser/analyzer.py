#===----------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# LLVM Advisor Analysis Engine - Comprehensive analysis of optimization
# remarks, profiling data, and performance insights.
#
#===----------------------------------------------------------------------===#

from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from .models import Project, RemarkType, OptimizationRemark
from .profile_parser import ProfileParser, ProfileData


class RemarkAnalyzer:
    def __init__(self, project: Project, profile_path: Optional[str] = None):
        self.project = project
        self.profile_data = None
        self.raw_trace_data = None
        if profile_path:
            parser = ProfileParser(profile_path)
            self.profile_data = parser.parse()
            self.raw_trace_data = parser.get_raw_data()
    
    def analyze_optimization_opportunities(self) -> Dict[str, Any]:
        missed_by_pass = defaultdict(list)
        
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if remark.remark_type == RemarkType.MISSED:
                    missed_by_pass[remark.pass_name].append(remark)
        
        opportunities = []
        for pass_name, remarks in missed_by_pass.items():
            # Count unique files affected by this pass
            unique_files = set()
            for remark in remarks:
                for unit in self.project.compilation_units:
                    if remark in unit.remarks:
                        unique_files.add(unit.source_file)
                        break
            
            opportunities.append({
                "pass": pass_name,
                "missed_count": len(remarks),
                "files_affected": len(unique_files),
                "impact": "high" if len(remarks) > 10 else "medium" if len(remarks) > 5 else "low"
            })
        
        return {
            "optimization_opportunities": sorted(opportunities, key=lambda x: x["missed_count"], reverse=True),
            "total_missed": sum(len(remarks) for remarks in missed_by_pass.values())
        }
    
    def analyze_performance_hotspots(self) -> Dict[str, Any]:
        function_remarks = defaultdict(list)
        
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                function_remarks[remark.function].append(remark)
        
        hotspots = []
        for function, remarks in function_remarks.items():
            if len(remarks) > 3:  # Functions with many remarks are potential hotspots
                passed = sum(1 for r in remarks if r.remark_type == RemarkType.PASSED)
                missed = sum(1 for r in remarks if r.remark_type == RemarkType.MISSED)
                
                hotspots.append({
                    "function": function,
                    "total_remarks": len(remarks),
                    "passed": passed,
                    "missed": missed,
                    "optimization_ratio": passed / len(remarks) if remarks else 0
                })
        
        return {
            "hotspots": sorted(hotspots, key=lambda x: x["total_remarks"], reverse=True)[:10],
            "total_functions_analyzed": len(function_remarks)
        }
    
    def analyze_offloading_efficiency(self) -> Dict[str, Any]:
        offloading_remarks = []
        
        for unit in self.project.compilation_units:
            offloading_remarks.extend(unit.get_offloading_remarks())
        
        if not offloading_remarks:
            return {"offloading_remarks": 0, "efficiency": "N/A"}
        
        passed = sum(1 for r in offloading_remarks if r.remark_type == RemarkType.PASSED)
        missed = sum(1 for r in offloading_remarks if r.remark_type == RemarkType.MISSED)
        
        efficiency = passed / len(offloading_remarks) if offloading_remarks else 0
        
        return {
            "offloading_remarks": len(offloading_remarks),
            "passed": passed,
            "missed": missed,
            "efficiency": efficiency,
            "efficiency_rating": "excellent" if efficiency > 0.8 else "good" if efficiency > 0.6 else "needs_improvement"
        }

    def analyze_profiling_data(self) -> Dict[str, Any]:
        """Analyze performance-related remarks for profiling insights"""
        static_analysis = {
            "loop_analysis": self._analyze_loops(),
            "vectorization_analysis": self._analyze_vectorization(),
            "inlining_analysis": self._analyze_inlining(),
            "memory_analysis": self._analyze_memory_operations(),
            "kernel_analysis": self._analyze_kernels(),
            "hotspot_files": self._analyze_file_hotspots()
        }
        
        if self.profile_data:
            runtime_analysis = self._analyze_runtime_profile()
            static_analysis.update(runtime_analysis)
        
        return static_analysis
    
    def analyze_optimization_insights(self) -> Dict[str, Any]:
        """Detailed optimization analysis for optimization tab"""
        return {
            "vectorization_opportunities": self._get_vectorization_opportunities(),
            "loop_optimization": self._analyze_loop_optimizations(),
            "function_optimization": self._analyze_function_optimizations(),
            "memory_optimization": self._analyze_memory_optimizations(),
            "parallelization_opportunities": self._get_parallelization_opportunities(),
            "compiler_recommendations": self._generate_compiler_recommendations()
        }
    
    def analyze_hardware_insights(self) -> Dict[str, Any]:
        """Hardware-specific analysis for hardware tab"""
        return {
            "gpu_utilization": self._analyze_gpu_utilization(),
            "memory_hierarchy": self._analyze_memory_hierarchy(),
            "compute_patterns": self._analyze_compute_patterns(),
            "offloading_patterns": self._analyze_offloading_patterns(),
            "architecture_recommendations": self._generate_architecture_recommendations()
        }
    
    def _analyze_loops(self) -> Dict[str, Any]:
        """Analyze loop-related remarks"""
        loop_remarks = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if any(keyword in remark.get_message().lower() for keyword in ['loop', 'unroll', 'vectorize']):
                    loop_remarks.append(remark)
        
        loop_stats = Counter([r.pass_name for r in loop_remarks])
        return {
            "total_loops": len(loop_remarks),
            "by_pass": dict(loop_stats),
            "vectorized": len([r for r in loop_remarks if 'vectorize' in r.pass_name and r.remark_type == RemarkType.PASSED]),
            "failed_vectorization": len([r for r in loop_remarks if 'vectorize' in r.pass_name and r.remark_type == RemarkType.MISSED])
        }
    
    def _analyze_vectorization(self) -> Dict[str, Any]:
        """Analyze vectorization performance"""
        vec_remarks = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if 'vectorize' in remark.pass_name.lower():
                    vec_remarks.append(remark)
        
        successful = [r for r in vec_remarks if r.remark_type == RemarkType.PASSED]
        failed = [r for r in vec_remarks if r.remark_type == RemarkType.MISSED]
        
        return {
            "total_vectorization_attempts": len(vec_remarks),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(vec_remarks) if vec_remarks else 0,
            "common_failures": self._get_common_failure_reasons(failed)
        }
    
    def _analyze_inlining(self) -> Dict[str, Any]:
        """Analyze function inlining performance"""
        inline_remarks = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if remark.pass_name == 'inline':
                    inline_remarks.append(remark)
        
        successful = [r for r in inline_remarks if r.remark_type == RemarkType.PASSED]
        failed = [r for r in inline_remarks if r.remark_type == RemarkType.MISSED]
        
        return {
            "total_inline_attempts": len(inline_remarks),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(inline_remarks) if inline_remarks else 0
        }
    
    def _analyze_memory_operations(self) -> Dict[str, Any]:
        """Analyze memory-related optimizations"""
        memory_remarks = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                message = remark.get_message().lower()
                if any(keyword in message for keyword in ['load', 'store', 'memory', 'cache', 'transfer']):
                    memory_remarks.append(remark)
        
        return {
            "total_memory_operations": len(memory_remarks),
            "optimized": len([r for r in memory_remarks if r.remark_type == RemarkType.PASSED]),
            "missed_optimizations": len([r for r in memory_remarks if r.remark_type == RemarkType.MISSED])
        }
    
    def _analyze_kernels(self) -> Dict[str, Any]:
        """Analyze GPU kernel performance from both remarks and profile data"""
        kernel_remarks = []
        kernel_functions = set()
        target_regions = set()
        
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                # Check multiple indicators for kernel functions
                is_kernel = (
                    '__omp_offloading_' in remark.function or 
                    'kernel' in remark.function.lower() or
                    remark.is_offloading_related()
                )
                
                if is_kernel:
                    kernel_remarks.append(remark)
                    
                    # Extract unique kernel functions
                    if '__omp_offloading_' in remark.function:
                        kernel_functions.add(remark.function)
                    elif remark.is_offloading_related():
                        # For OpenMP optimization remarks, create unique identifiers
                        if remark.debug_loc:
                            # Use file:line as unique kernel identifier for target regions
                            kernel_id = f"{remark.debug_loc.file}:{remark.debug_loc.line}"
                            target_regions.add(kernel_id)
                        else:
                            # Fallback to function name
                            kernel_functions.add(remark.function)
        
        # Count from static analysis of source files for target directives
        static_kernel_count = 0
        for unit in self.project.compilation_units:
            source_lines = unit.get_source_lines(".")
            if source_lines:
                for line in source_lines:
                    # Count OpenMP target directives directly from source
                    if '#pragma omp target' in line and not line.strip().startswith('//'):
                        static_kernel_count += 1
        
        # Get accurate kernel count from profile data if available
        profile_kernels = 0
        if self.raw_trace_data and 'traceEvents' in self.raw_trace_data:
            # Primary method: Get count from "Total Runtime: target exe" summary event
            summary_events = [e for e in self.raw_trace_data['traceEvents'] 
                            if (e.get('name', '') == 'Total Runtime: target exe' and 
                                e.get('args', {}).get('count') is not None)]
            
            if summary_events:
                # Use the count from summary which represents actual kernel executions
                profile_kernels = summary_events[0]['args']['count']
            else:
                # Fallback: Count individual kernel execution events
                kernel_events = [e for e in self.raw_trace_data['traceEvents'] 
                               if (e.get('name', '') == 'Runtime: target exe' and 
                                   e.get('ph') == 'X' and e.get('dur', 0) > 0)]
                profile_kernels = len(kernel_events)
                
                # Another fallback: Look for other kernel execution patterns
                if profile_kernels == 0:
                    target_events = [e for e in self.raw_trace_data['traceEvents'] 
                                   if (e.get('name', '').startswith('Kernel Target') and 
                                       e.get('ph') == 'X')]
                    profile_kernels = len(target_events)
        
        elif self.profile_data and hasattr(self.profile_data, 'kernels'):
            profile_kernels = len(self.profile_data.kernels)
        
        # Prioritize profile data when available, otherwise use static analysis
        detected_kernels = profile_kernels if profile_kernels > 0 else max(
            len(kernel_functions),
            len(target_regions), 
            static_kernel_count
        )
        
        return {
            "total_kernels": detected_kernels,
            "total_kernel_remarks": len(kernel_remarks),
            "optimized_kernels": len([f for f in kernel_functions if any(
                r.remark_type == RemarkType.PASSED for r in kernel_remarks if r.function == f
            )]),
            "profile_detected_kernels": profile_kernels,
            "remarks_detected_kernels": len(kernel_functions),
            "target_regions_detected": len(target_regions),
            "static_analysis_kernels": static_kernel_count
        }
    
    def _analyze_file_hotspots(self) -> List[Dict[str, Any]]:
        """Identify files with most optimization activity"""
        file_stats = []
        for unit in self.project.compilation_units:
            passed = len([r for r in unit.remarks if r.remark_type == RemarkType.PASSED])
            missed = len([r for r in unit.remarks if r.remark_type == RemarkType.MISSED])
            total = len(unit.remarks)
            
            if total > 0:
                file_stats.append({
                    "file": unit.source_file,
                    "total_remarks": total,
                    "passed": passed,
                    "missed": missed,
                    "optimization_ratio": passed / total,
                    "activity_score": total
                })
        
        return sorted(file_stats, key=lambda x: x["activity_score"], reverse=True)[:10]
    
    def _get_vectorization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify missed vectorization opportunities"""
        opportunities = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if ('vectorize' in remark.pass_name.lower() and 
                    remark.remark_type == RemarkType.MISSED):
                    opportunities.append({
                        "file": unit.source_file,
                        "line": remark.debug_loc.line if remark.debug_loc else 0,
                        "function": remark.function,
                        "reason": remark.get_message(),
                        "pass": remark.pass_name
                    })
        return opportunities[:20]  # Top 20 opportunities
    
    def _analyze_loop_optimizations(self) -> Dict[str, Any]:
        """Analyze loop optimization patterns"""
        loop_passes = ['loop-vectorize', 'loop-unroll', 'licm', 'loop-idiom']
        loop_data = {}
        
        for pass_name in loop_passes:
            remarks = []
            for unit in self.project.compilation_units:
                for remark in unit.remarks:
                    if remark.pass_name == pass_name:
                        remarks.append(remark)
            
            loop_data[pass_name] = {
                "total": len(remarks),
                "passed": len([r for r in remarks if r.remark_type == RemarkType.PASSED]),
                "missed": len([r for r in remarks if r.remark_type == RemarkType.MISSED])
            }
        
        return loop_data
    
    def _analyze_function_optimizations(self) -> Dict[str, Any]:
        """Analyze function-level optimizations"""
        function_passes = ['inline', 'dce', 'dse', 'gvn']
        func_data = {}
        
        for pass_name in function_passes:
            remarks = []
            for unit in self.project.compilation_units:
                for remark in unit.remarks:
                    if remark.pass_name == pass_name:
                        remarks.append(remark)
            
            func_data[pass_name] = {
                "total": len(remarks),
                "passed": len([r for r in remarks if r.remark_type == RemarkType.PASSED]),
                "missed": len([r for r in remarks if r.remark_type == RemarkType.MISSED])
            }
        
        return func_data
    
    def _analyze_memory_optimizations(self) -> List[Dict[str, Any]]:
        """Analyze memory optimization opportunities"""
        memory_issues = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if (remark.remark_type == RemarkType.MISSED and
                    any(keyword in remark.get_message().lower() for keyword in 
                        ['load', 'store', 'memory', 'cache', 'clobbered'])):
                    memory_issues.append({
                        "file": unit.source_file,
                        "line": remark.debug_loc.line if remark.debug_loc else 0,
                        "function": remark.function,
                        "issue": remark.get_message(),
                        "pass": remark.pass_name
                    })
        return memory_issues[:15]
    
    def _get_parallelization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify parallelization opportunities"""
        opportunities = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if (any(keyword in remark.get_message().lower() for keyword in 
                       ['parallel', 'thread', 'openmp', 'offload']) and
                    remark.remark_type == RemarkType.MISSED):
                    opportunities.append({
                        "file": unit.source_file,
                        "line": remark.debug_loc.line if remark.debug_loc else 0,
                        "function": remark.function,
                        "opportunity": remark.get_message(),
                        "pass": remark.pass_name
                    })
        return opportunities[:10]
    
    def _generate_compiler_recommendations(self) -> List[str]:
        """Generate compiler optimization recommendations"""
        recommendations = []
        
        # Analyze common missed optimizations
        missed_passes = Counter()
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if remark.remark_type == RemarkType.MISSED:
                    missed_passes[remark.pass_name] += 1
        
        if missed_passes.get('loop-vectorize', 0) > 10:
            recommendations.append("Consider using -ffast-math for aggressive vectorization")
        if missed_passes.get('inline', 0) > 20:
            recommendations.append("Increase inlining thresholds with -finline-limit")
        if missed_passes.get('gvn', 0) > 5:
            recommendations.append("Enable aggressive optimization with -O3")
        
        return recommendations
    
    def _analyze_gpu_utilization(self) -> Dict[str, Any]:
        """Analyze GPU utilization patterns"""
        gpu_remarks = []
        for unit in self.project.compilation_units:
            gpu_remarks.extend(unit.get_offloading_remarks())
        
        kernel_functions = set(r.function for r in gpu_remarks if '__omp_offloading_' in r.function)
        return {
            "total_gpu_functions": len(kernel_functions),
            "optimization_coverage": len([r for r in gpu_remarks if r.remark_type == RemarkType.PASSED]) / len(gpu_remarks) if gpu_remarks else 0,
            "offloading_efficiency": self.analyze_offloading_efficiency()
        }
    
    def _analyze_memory_hierarchy(self) -> Dict[str, Any]:
        """Analyze memory access patterns for GPU"""
        memory_remarks = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if (remark.is_offloading_related() and
                    any(keyword in remark.get_message().lower() for keyword in 
                        ['memory', 'load', 'store', 'cache', 'shared', 'global'])):
                    memory_remarks.append(remark)
        
        return {
            "memory_operations": len(memory_remarks),
            "optimized_memory": len([r for r in memory_remarks if r.remark_type == RemarkType.PASSED]),
            "memory_issues": len([r for r in memory_remarks if r.remark_type == RemarkType.MISSED])
        }
    
    def _analyze_compute_patterns(self) -> Dict[str, Any]:
        """Analyze compute utilization patterns"""
        compute_remarks = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if (remark.is_offloading_related() and
                    any(keyword in remark.get_message().lower() for keyword in 
                        ['compute', 'thread', 'warp', 'simd', 'vector'])):
                    compute_remarks.append(remark)
        
        return {
            "compute_operations": len(compute_remarks),
            "compute_efficiency": len([r for r in compute_remarks if r.remark_type == RemarkType.PASSED]) / len(compute_remarks) if compute_remarks else 0
        }
    
    def _analyze_offloading_patterns(self) -> Dict[str, Any]:
        """Analyze offloading patterns and effectiveness"""
        offload_remarks = []
        for unit in self.project.compilation_units:
            offload_remarks.extend(unit.get_offloading_remarks())
        
        offload_functions = Counter(r.function for r in offload_remarks if '__omp_offloading_' in r.function)
        return {
            "offloaded_functions": len(offload_functions),
            "most_active_kernels": dict(offload_functions.most_common(5)),
            "offloading_success_rate": len([r for r in offload_remarks if r.remark_type == RemarkType.PASSED]) / len(offload_remarks) if offload_remarks else 0
        }
    
    def _generate_architecture_recommendations(self) -> List[str]:
        """Generate architecture-specific recommendations"""
        recommendations = []
        
        offload_remarks = []
        for unit in self.project.compilation_units:
            offload_remarks.extend(unit.get_offloading_remarks())
        
        if len(offload_remarks) > 0:
            efficiency = len([r for r in offload_remarks if r.remark_type == RemarkType.PASSED]) / len(offload_remarks)
            if efficiency < 0.7:
                recommendations.append("Consider optimizing GPU memory access patterns")
                recommendations.append("Review data transfer between host and device")
        
        vectorization_remarks = []
        for unit in self.project.compilation_units:
            for remark in unit.remarks:
                if 'vectorize' in remark.pass_name:
                    vectorization_remarks.append(remark)
        
        if len(vectorization_remarks) > 0:
            vec_efficiency = len([r for r in vectorization_remarks if r.remark_type == RemarkType.PASSED]) / len(vectorization_remarks)
            if vec_efficiency < 0.5:
                recommendations.append("Consider SIMD-friendly data structures")
        
        return recommendations
    
    def _get_common_failure_reasons(self, failed_remarks: List[OptimizationRemark]) -> List[str]:
        """Extract common failure reasons from remarks"""
        reasons = Counter()
        for remark in failed_remarks:
            message = remark.get_message().lower()
            if 'uncountable' in message:
                reasons['Uncountable loops'] += 1
            elif 'definition unavailable' in message:
                reasons['Function definition unavailable'] += 1
            elif 'clobbered' in message:
                reasons['Memory dependencies'] += 1
            elif 'impossible' in message:
                reasons['Vectorization impossible'] += 1
            else:
                reasons['Other'] += 1
        
        return [f"{reason}: {count}" for reason, count in reasons.most_common(5)]
    
    def _analyze_runtime_profile(self) -> Dict[str, Any]:
        """Analyze runtime profiling data from LIBOMPTARGET_PROFILE"""
        if not self.profile_data and not self.raw_trace_data:
            return {}
        
        # Prioritize raw Chrome Trace format data
        if self.raw_trace_data:
            return {
                "trace_data": self.raw_trace_data,
                "performance_bottlenecks": self._identify_trace_bottlenecks(self.raw_trace_data),
                "optimization_recommendations": self._generate_trace_recommendations(self.raw_trace_data)
            }
        
        # Handle Chrome Trace format from profile_data
        if hasattr(self.profile_data, 'trace_events') or isinstance(self.profile_data, dict):
            trace_data = self.profile_data if isinstance(self.profile_data, dict) else self.profile_data.__dict__
            
            return {
                "trace_data": trace_data,
                "performance_bottlenecks": self._identify_trace_bottlenecks(trace_data),
                "optimization_recommendations": self._generate_trace_recommendations(trace_data)
            }
        
        # Fallback to original ProfileData structure (if implemented)
        if self.profile_data and hasattr(self.profile_data, 'total_time'):
            return {
                "runtime_performance": {
                    "total_execution_time_us": self.profile_data.total_time,
                    "device_time_us": self.profile_data.device_time,
                    "host_time_us": self.profile_data.host_time,
                    "memory_transfer_time_us": self.profile_data.memory_transfer_time,
                    "device_utilization_percent": (self.profile_data.device_time / self.profile_data.total_time * 100) if self.profile_data.total_time > 0 else 0
                },
                "kernel_performance": [
                    {
                        "name": kernel.name,
                        "execution_time_us": kernel.execution_time,
                        "launch_time_us": kernel.launch_time,
                        "device_id": kernel.device_id,
                        "grid_size": kernel.grid_size,
                        "block_size": kernel.block_size
                    } for kernel in self.profile_data.kernels
                ],
                "performance_bottlenecks": self._identify_performance_bottlenecks(),
                "optimization_recommendations": self._generate_runtime_recommendations()
            }
        
        return {}

    def _identify_trace_bottlenecks(self, trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from Chrome Trace format data"""
        if not trace_data or 'traceEvents' not in trace_data:
            return []
        
        bottlenecks = []
        events = [e for e in trace_data['traceEvents'] if e.get('ph') == 'X' and e.get('dur')]
        
        if not events:
            return bottlenecks
        
        # Calculate time distributions
        kernel_events = [e for e in events if 'target exe' in e.get('name', '')]
        memory_events = [e for e in events if any(term in e.get('name', '') for term in ['HostToDev', 'DevToHost'])]
        init_events = [e for e in events if 'init' in e.get('name', '').lower()]
        
        total_time = max(e['ts'] + e['dur'] for e in events) - min(e['ts'] for e in events)
        kernel_time = sum(e['dur'] for e in kernel_events)
        memory_time = sum(e['dur'] for e in memory_events)
        init_time = sum(e['dur'] for e in init_events)
        
        # Memory transfer bottleneck
        if memory_time > kernel_time * 0.5:
            bottlenecks.append({
                "type": "memory_transfer",
                "severity": "high",
                "description": "Memory transfers take significant time compared to kernel execution",
                "impact_percent": (memory_time / total_time) * 100
            })
        
        # Initialization overhead
        if init_time > total_time * 0.3:
            bottlenecks.append({
                "type": "initialization_overhead",
                "severity": "medium", 
                "description": "Initialization takes significant portion of total execution time",
                "impact_percent": (init_time / total_time) * 100
            })
        
        # Kernel utilization
        kernel_util = (kernel_time / total_time) * 100 if total_time > 0 else 0
        if kernel_util < 30:
            bottlenecks.append({
                "type": "low_kernel_utilization",
                "severity": "medium",
                "description": f"Kernel utilization is only {kernel_util:.1f}%",
                "impact_percent": 100 - kernel_util
            })
        
        # Kernel execution imbalance
        if len(kernel_events) > 1:
            durations = [e['dur'] for e in kernel_events]
            max_dur = max(durations)
            min_dur = min(durations)
            if max_dur > min_dur * 3:  # 3x difference indicates imbalance
                bottlenecks.append({
                    "type": "kernel_imbalance",
                    "severity": "low",
                    "description": "Significant execution time variance between kernel launches",
                    "impact_percent": ((max_dur - min_dur) / max_dur) * 100
                })
        
        return bottlenecks

    def _generate_trace_recommendations(self, trace_data: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on Chrome Trace format data"""
        if not trace_data or 'traceEvents' not in trace_data:
            return []
        
        recommendations = []
        events = [e for e in trace_data['traceEvents'] if e.get('ph') == 'X' and e.get('dur')]
        
        if not events:
            return recommendations
        
        kernel_events = [e for e in events if 'target exe' in e.get('name', '')]
        memory_events = [e for e in events if any(term in e.get('name', '') for term in ['HostToDev', 'DevToHost'])]
        
        total_time = max(e['ts'] + e['dur'] for e in events) - min(e['ts'] for e in events)
        kernel_time = sum(e['dur'] for e in kernel_events)
        memory_time = sum(e['dur'] for e in memory_events)
        
        # Memory transfer recommendations
        if memory_time > kernel_time * 0.3:
            recommendations.append("Consider optimizing data transfers - use unified memory or asynchronous transfers")
            recommendations.append("Reduce data movement by keeping data on device between kernel launches")
        
        # Kernel execution recommendations
        if len(kernel_events) > 5:
            avg_kernel_time = kernel_time / len(kernel_events)
            if avg_kernel_time < 100:  # Very short kernels (< 100μs)
                recommendations.append("Consider kernel fusion to reduce launch overhead")
        
        # Device utilization recommendations
        kernel_util = (kernel_time / total_time) * 100 if total_time > 0 else 0
        if kernel_util < 50:
            recommendations.append("Increase workload size or use multiple devices to improve utilization")
            recommendations.append("Consider overlapping computation with data transfers")
        
        # Specific kernel analysis
        if kernel_events:
            for event in kernel_events:
                detail = event.get('args', {}).get('detail', '')
                if 'NumTeams=0' in detail:
                    recommendations.append("Some kernels launch with NumTeams=0 - verify OpenMP target directives")
        
        return recommendations
    
    def _identify_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from runtime data"""
        if not self.profile_data:
            return []
        
        bottlenecks = []
        
        # Memory transfer bottleneck
        if self.profile_data.memory_transfer_time > self.profile_data.device_time * 0.5:
            bottlenecks.append({
                "type": "memory_transfer",
                "severity": "high",
                "description": "Memory transfers take significant time compared to computation",
                "impact_percent": (self.profile_data.memory_transfer_time / self.profile_data.total_time) * 100
            })
        
        # Low device utilization
        device_util = (self.profile_data.device_time / self.profile_data.total_time) * 100 if self.profile_data.total_time > 0 else 0
        if device_util < 50:
            bottlenecks.append({
                "type": "low_device_utilization",
                "severity": "medium",
                "description": f"Device utilization is only {device_util:.1f}%",
                "impact_percent": 100 - device_util
            })
        
        # Kernel execution imbalance
        if len(self.profile_data.kernels) > 1:
            execution_times = [k.execution_time for k in self.profile_data.kernels]
            max_time = max(execution_times)
            min_time = min(execution_times)
            if max_time > min_time * 5:  # 5x difference indicates imbalance
                bottlenecks.append({
                    "type": "kernel_imbalance",
                    "severity": "medium",
                    "description": "Significant execution time variance between kernels",
                    "impact_percent": ((max_time - min_time) / max_time) * 100
                })
        
        return bottlenecks
    
    def _generate_runtime_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on runtime data"""
        if not self.profile_data:
            return []
        
        recommendations = []
        
        # Memory transfer optimization
        if self.profile_data.memory_transfer_time > self.profile_data.device_time * 0.3:
            recommendations.append("Consider data layout optimizations to reduce memory transfers")
            recommendations.append("Use unified memory or async transfers where possible")
        
        # Device utilization
        device_util = (self.profile_data.device_time / self.profile_data.total_time) * 100 if self.profile_data.total_time > 0 else 0
        if device_util < 70:
            recommendations.append("Increase workload size or use multiple kernels to improve device utilization")
        
        # Kernel optimization
        for kernel in self.profile_data.kernels:
            if kernel.execution_time < 1000:  # Less than 1ms
                recommendations.append(f"Kernel '{kernel.name}' has very short execution time - consider kernel fusion")
        
        return recommendations
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        return {
            "project_summary": self.project.get_project_summary(),
            "optimization_opportunities": self.analyze_optimization_opportunities(),
            "performance_hotspots": self.analyze_performance_hotspots(),
            "offloading_efficiency": self.analyze_offloading_efficiency(),
            "profiling_data": self.analyze_profiling_data(),
            "optimization_insights": self.analyze_optimization_insights(),
            "hardware_insights": self.analyze_hardware_insights()
        }
