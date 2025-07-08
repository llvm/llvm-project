#===----------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# LLVM Advisor Data Models - Structured data models for optimization
# analysis, remarks, and profiling information.
#
#===----------------------------------------------------------------------===#

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path


class RemarkType(Enum):
    PASSED = "Passed"
    MISSED = "Missed" 
    ANALYSIS = "Analysis"


@dataclass
class DebugLocation:
    file: str
    line: int
    column: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['DebugLocation']:
        return cls(
            file=data.get('File', '<unknown>'),
            line=int(data.get('Line', 0)),
            column=int(data.get('Column', 0))
        ) if data else None


@dataclass
class RemarkArgument:
    type: str
    value: Any
    debug_loc: Optional[DebugLocation] = None
    
    @classmethod
    def from_dict(cls, data: Union[str, Dict[str, Any]]) -> 'RemarkArgument':
        if isinstance(data, str):
            return cls(type="String", value=data)
        
        # Consolidated argument type mapping
        type_mappings = {
            # Basic types
            "String": "String",
            "DebugLoc": None,  # Special handling
            # Function/code references
            **{key: key for key in ["Callee", "Caller", "Function", "BasicBlock", "Inst", 
                                   "Call", "OtherAccess", "ClobberedBy", "InfavorOfValue", "Reason"]},
            # Metrics and counts
            **{key: key for key in ["Type", "Cost", "Threshold", "VectorizationFactor", 
                                   "InterleaveCount", "NumVRCopies", "TotalCopiesCost",
                                   "NumStackBytes", "NumInstructions", "Line", "Column"]},
            # OpenMP/GPU specific
            **{key: key for key in ["ExternalNotKernel", "GlobalsSize", "LocalVarSize", 
                                   "NumRegions", "RegisterPressurerValue", "AssumedAddressSpace",
                                   "SPMDCompatibilityTracker", "GlobalizationLevel", "AddressSpace"]}
        }
        
        arg_type = None
        value = None
        debug_loc = None
        
        for key, val in data.items():
            if key == "DebugLoc":
                debug_loc = DebugLocation.from_dict(val)
            elif key in type_mappings:
                arg_type = type_mappings[key]
                value = val
                break
        
        return cls(type=arg_type or "Unknown", value=value, debug_loc=debug_loc)


@dataclass 
class OptimizationRemark:
    remark_type: RemarkType
    pass_name: str
    remark_name: str
    function: str
    debug_loc: Optional[DebugLocation]
    args: List[RemarkArgument]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationRemark':
        remark_type = {
            'Passed': RemarkType.PASSED,
            'Missed': RemarkType.MISSED,
            'Analysis': RemarkType.ANALYSIS
        }.get(data.get('_remark_type'), RemarkType.MISSED)
        
        return cls(
            remark_type=remark_type,
            pass_name=data.get('Pass', 'unknown'),
            remark_name=data.get('Name', 'unknown'),
            function=data.get('Function', 'unknown'),
            debug_loc=DebugLocation.from_dict(data.get('DebugLoc', {})),
            args=[RemarkArgument.from_dict(arg) for arg in data.get('Args', [])]
        )
    
    def get_message(self) -> str:
        parts = []
        for arg in self.args:
            if arg.type == "String":
                parts.append(str(arg.value))
            else:
                parts.append(f"[{arg.type}: {arg.value}]")
        return "".join(parts)
    
    def is_offloading_related(self) -> bool:
        """Enhanced detection of OpenMP offloading and GPU-related remarks"""
        offloading_indicators = [
            "__omp_offloading", "omp_outlined", "target", "kernel-info", "offload",
            "gpu", "cuda", "opencl", "sycl", "hip", "device", "host",
            "ExternalNotKernel", "SPMDCompatibilityTracker", "GlobalizationLevel",
            "NumRegions", "AddressSpace", "AssumedAddressSpace"
        ]
        
        # OpenMP-specific pass names that indicate offloading
        openmp_passes = [
            "openmp-opt", "kernel-info", "openmp", "target-region", 
            "offload", "device-lower", "gpu-lower"
        ]
        
        # Check function name, pass name, and remark name
        text_to_check = f"{self.function} {self.pass_name} {self.remark_name}".lower()
        if any(indicator in text_to_check for indicator in offloading_indicators):
            return True
            
        # Check if this is an OpenMP-related pass
        if any(omp_pass in self.pass_name.lower() for omp_pass in openmp_passes):
            return True
        
        # Check argument values for GPU/offloading specific content
        for arg in self.args:
            if hasattr(arg, 'value') and arg.value:
                arg_text = str(arg.value).lower()
                if any(indicator in arg_text for indicator in offloading_indicators):
                    return True
            
            # Check for specific OpenMP argument types
            if hasattr(arg, 'type') and arg.type in [
                "ExternalNotKernel", "SPMDCompatibilityTracker", "GlobalizationLevel",
                "NumRegions", "AddressSpace", "AssumedAddressSpace"
            ]:
                return True
        
        # Check message content for OpenMP target-related keywords
        message = self.get_message().lower()
        target_keywords = [
            "target region", "target directive", "offload", "device", 
            "kernel", "gpu", "accelerator", "teams", "thread_limit"
        ]
        if any(keyword in message for keyword in target_keywords):
            return True
        
        return False


@dataclass
class CompilationUnit:
    source_file: str
    remarks: List[OptimizationRemark] = field(default_factory=list)
    build_time: Optional[float] = None
    optimization_level: Optional[str] = None
    target_arch: Optional[str] = None
    
    def get_remarks_by_pass(self) -> Dict[str, List[OptimizationRemark]]:
        result = {}
        for remark in self.remarks:
            result.setdefault(remark.pass_name, []).append(remark)
        return result
    
    def get_remarks_by_type(self) -> Dict[RemarkType, List[OptimizationRemark]]:
        result = {t: [] for t in RemarkType}
        for remark in self.remarks:
            result[remark.remark_type].append(remark)
        return result
    
    def get_offloading_remarks(self) -> List[OptimizationRemark]:
        return [r for r in self.remarks if r.is_offloading_related()]
    
    def get_summary(self) -> Dict[str, Any]:
        by_type = self.get_remarks_by_type()
        return {
            "source_file": self.source_file,
            "total_remarks": len(self.remarks),
            "passed": len(by_type[RemarkType.PASSED]),
            "missed": len(by_type[RemarkType.MISSED]),
            "analysis": len(by_type[RemarkType.ANALYSIS]),
            "offloading_remarks": len(self.get_offloading_remarks()),
            "passes_involved": list(self.get_remarks_by_pass().keys()),
            "build_time": self.build_time,
            "optimization_level": self.optimization_level,
            "target_arch": self.target_arch
        }
    
    def deduplicate_remarks(self):
        """Remove duplicate remarks based on key attributes."""
        seen = set()
        unique_remarks = []
        
        for remark in self.remarks:
            # Create a key based on remark attributes
            key = (
                remark.remark_type,
                remark.pass_name,
                remark.remark_name,
                remark.function,
                remark.debug_loc.line if remark.debug_loc else 0,
                remark.debug_loc.column if remark.debug_loc else 0
            )
            
            if key not in seen:
                seen.add(key)
                unique_remarks.append(remark)
        
        self.remarks = unique_remarks

    def load_source_code(self, source_directory: str = ".") -> Optional[str]:
        """Load the actual source code for this file."""
        possible_paths = [
            Path(source_directory) / self.source_file,
            Path(source_directory) / "src" / self.source_file,
            Path(source_directory) / "source" / self.source_file,
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception:
                    continue
        return None
    
    def get_source_lines(self, source_directory: str = ".") -> List[str]:
        """Get source code as list of lines."""
        source_code = self.load_source_code(source_directory)
        return source_code.split('\n') if source_code else []


@dataclass
class Project:
    name: str
    compilation_units: List[CompilationUnit] = field(default_factory=list)
    build_system: Optional[str] = None
    
    def add_compilation_unit(self, unit: CompilationUnit):
        self.compilation_units.append(unit)
    
    def get_project_summary(self) -> Dict[str, Any]:
        total_remarks = sum(len(unit.remarks) for unit in self.compilation_units)
        total_offloading = sum(len(unit.get_offloading_remarks()) for unit in self.compilation_units)
        all_passes = set()
        
        # Collect type counts across all units
        type_counts = {remark_type: 0 for remark_type in RemarkType}
        pass_counts = {}
        
        for unit in self.compilation_units:
            all_passes.update(unit.get_remarks_by_pass().keys())
            by_type = unit.get_remarks_by_type()
            for remark_type, remarks in by_type.items():
                type_counts[remark_type] += len(remarks)
            
            for remark in unit.remarks:
                pass_counts[remark.pass_name] = pass_counts.get(remark.pass_name, 0) + 1
        
        return {
            "project_name": self.name,
            "compilation_units": len(self.compilation_units),
            "total_files": len(self.compilation_units),  # Added for frontend compatibility
            "total_remarks": total_remarks,
            "total_offloading_remarks": total_offloading,
            "total_passed": type_counts[RemarkType.PASSED],
            "total_missed": type_counts[RemarkType.MISSED], 
            "total_analysis": type_counts[RemarkType.ANALYSIS],
            "unique_passes": list(all_passes),
            "pass_counts": pass_counts,
            "most_active_passes": sorted(pass_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "build_system": self.build_system
        }
    
    def get_file_list(self) -> List[Dict[str, Any]]:
        files = []
        for unit in self.compilation_units:
            summary = unit.get_summary()
            files.append({
                "file": unit.source_file,
                "remarks": summary["total_remarks"],
                "passed": summary["passed"],
                "missed": summary["missed"],
                "offloading": summary["offloading_remarks"]
            })
        return files
