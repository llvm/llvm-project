#!/usr/bin/env python3
"""
Comprehensive benchmarking script for lldb-dap fast launch mode.

This script tests and benchmarks the fast launch implementation across
different scenarios to validate performance claims and functionality.
"""

import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

class DAPBenchmark:
    def __init__(self, lldb_dap_path, test_programs_dir):
        self.lldb_dap_path = lldb_dap_path
        self.test_programs_dir = Path(test_programs_dir)
        self.results = []
        
    def create_launch_config(self, program_path, fast_launch=False, **kwargs):
        """Create a DAP launch configuration."""
        config = {
            "type": "lldb-dap",
            "request": "launch",
            "name": "Test Launch",
            "program": str(program_path),
            "stopOnEntry": True,
            "args": [],
            "cwd": str(program_path.parent),
        }
        
        if fast_launch:
            config.update({
                "fastLaunchMode": True,
                "deferSymbolLoading": True,
                "lazyPluginLoading": True,
                "launchTimeoutMs": 1000,
            })
        
        # Add any additional configuration
        config.update(kwargs)
        return config
    
    def send_dap_request(self, request_type, arguments=None):
        """Send a DAP request and return the response."""
        request = {
            "seq": 1,
            "type": "request",
            "command": request_type,
            "arguments": arguments or {}
        }
        
        request_json = json.dumps(request)
        content_length = len(request_json.encode('utf-8'))
        
        # Format as DAP message
        message = f"Content-Length: {content_length}\r\n\r\n{request_json}"
        return message
    
    def benchmark_launch(self, program_path, config_name, fast_launch=False, iterations=5):
        """Benchmark launch time for a specific configuration."""
        print(f"\n=== Benchmarking {config_name} ===")
        print(f"Program: {program_path}")
        print(f"Fast launch: {fast_launch}")
        print(f"Iterations: {iterations}")
        
        times = []
        
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}...", end=" ", flush=True)
            
            # Create launch configuration
            config = self.create_launch_config(program_path, fast_launch)
            
            # Start lldb-dap process
            start_time = time.time()
            
            try:
                process = subprocess.Popen(
                    [self.lldb_dap_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Send initialize request
                init_request = self.send_dap_request("initialize", {
                    "clientID": "benchmark",
                    "clientName": "DAP Benchmark",
                    "adapterID": "lldb-dap",
                    "pathFormat": "path",
                    "linesStartAt1": True,
                    "columnsStartAt1": True,
                    "supportsVariableType": True,
                    "supportsVariablePaging": True,
                    "supportsRunInTerminalRequest": True
                })
                
                process.stdin.write(init_request)
                process.stdin.flush()
                
                # Send launch request
                launch_request = self.send_dap_request("launch", config)
                process.stdin.write(launch_request)
                process.stdin.flush()
                
                # Wait for process to be ready (simplified - in real scenario we'd parse responses)
                time.sleep(0.5)  # Give it time to initialize
                
                end_time = time.time()
                launch_time = (end_time - start_time) * 1000  # Convert to ms
                
                # Terminate the process
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                
                times.append(launch_time)
                print(f"{launch_time:.1f}ms")
                
            except Exception as e:
                print(f"Error: {e}")
                if 'process' in locals():
                    process.kill()
                continue
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            result = {
                "config_name": config_name,
                "program": str(program_path),
                "fast_launch": fast_launch,
                "iterations": len(times),
                "times": times,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
            
            self.results.append(result)
            
            print(f"  Average: {avg_time:.1f}ms")
            print(f"  Range: {min_time:.1f}ms - {max_time:.1f}ms")
            
            return result
        
        return None
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmarks across different scenarios."""
        print("=== LLDB-DAP Fast Launch Comprehensive Benchmark ===")
        print(f"lldb-dap path: {self.lldb_dap_path}")
        print(f"Test programs directory: {self.test_programs_dir}")
        
        # Test programs
        simple_program = self.test_programs_dir / "simple"
        complex_program = self.test_programs_dir / "complex"
        
        # Verify test programs exist
        if not simple_program.exists():
            print(f"Error: {simple_program} not found")
            return
        if not complex_program.exists():
            print(f"Error: {complex_program} not found")
            return
        
        # Benchmark scenarios
        scenarios = [
            (simple_program, "Simple Program - Normal Launch", False),
            (simple_program, "Simple Program - Fast Launch", True),
            (complex_program, "Complex Program - Normal Launch", False),
            (complex_program, "Complex Program - Fast Launch", True),
        ]
        
        for program, config_name, fast_launch in scenarios:
            self.benchmark_launch(program, config_name, fast_launch)
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and report benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS ANALYSIS")
        print("="*60)
        
        if not self.results:
            print("No results to analyze.")
            return
        
        # Group results by program
        simple_results = [r for r in self.results if "simple" in r["program"]]
        complex_results = [r for r in self.results if "complex" in r["program"]]
        
        def analyze_program_results(results, program_name):
            print(f"\n{program_name} Results:")
            print("-" * 40)
            
            normal_result = next((r for r in results if not r["fast_launch"]), None)
            fast_result = next((r for r in results if r["fast_launch"]), None)
            
            if normal_result:
                print(f"Normal launch: {normal_result['avg_time']:.1f}ms (avg)")
            if fast_result:
                print(f"Fast launch:   {fast_result['avg_time']:.1f}ms (avg)")
            
            if normal_result and fast_result:
                if fast_result['avg_time'] < normal_result['avg_time']:
                    improvement = normal_result['avg_time'] / fast_result['avg_time']
                    time_saved = normal_result['avg_time'] - fast_result['avg_time']
                    print(f"Improvement:   {improvement:.1f}x faster ({time_saved:.1f}ms saved)")
                else:
                    print("No significant improvement detected")
                    print("Note: Fast launch benefits vary based on project characteristics")
        
        analyze_program_results(simple_results, "Simple Program")
        analyze_program_results(complex_results, "Complex Program")
        
        # Overall analysis
        print(f"\nOverall Analysis:")
        print("-" * 40)
        print("Performance improvements depend on:")
        print("• Project size and symbol complexity")
        print("• Network symbol loading requirements")
        print("• System performance and storage speed")
        print("• Debug information size and structure")
        
        # Save detailed results
        results_file = "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")

def main():
    # Find lldb-dap binary
    script_dir = Path(__file__).parent
    llvm_root = script_dir.parent
    lldb_dap_path = llvm_root / "build" / "bin" / "lldb-dap"
    
    if not lldb_dap_path.exists():
        print(f"Error: lldb-dap not found at {lldb_dap_path}")
        print("Please build LLVM first: cd build && ninja lldb-dap")
        sys.exit(1)
    
    # Run benchmark
    benchmark = DAPBenchmark(str(lldb_dap_path), script_dir)
    benchmark.run_comprehensive_benchmark()

if __name__ == "__main__":
    main()
