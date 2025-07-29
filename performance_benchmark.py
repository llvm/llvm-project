#!/usr/bin/env python3
"""
Performance benchmark script to measure lldb-dap startup times vs other debuggers.
This helps identify the exact bottlenecks causing the 3000ms vs 120-400ms
performance gap.
"""

import subprocess
import time
import json
import os
import sys
from pathlib import Path

def create_test_program():
    """Create a simple test program for debugging."""
    test_program = Path("test_performance.c")
    test_program.write_text("""
#include <stdio.h>
int main() {
    printf("Hello, World!\\n");
    return 0;
}
""")
    
    # Compile with debug info
    subprocess.run(["clang", "-g", "-o", "test_performance",
                    "test_performance.c"], check=True)
    return Path("test_performance").absolute()

def create_dap_message(command, arguments=None):
    """Create a DAP protocol message."""
    if arguments is None:
        arguments = {}
    
    message = {
        "seq": 1,
        "type": "request", 
        "command": command,
        "arguments": arguments
    }
    
    content = json.dumps(message)
    return f"Content-Length: {len(content)}\r\n\r\n{content}"

def benchmark_lldb_dap_normal(program_path):
    """Benchmark normal lldb-dap startup."""
    print("=== Benchmarking Normal LLDB-DAP ===")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            ["./build/bin/lldb-dap"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send initialize request
        init_msg = create_dap_message("initialize", {
            "clientID": "benchmark",
            "adapterID": "lldb-dap"
        })
        
        # Send launch request
        launch_msg = create_dap_message("launch", {
            "program": str(program_path),
            "stopOnEntry": True
        })
        
        process.stdin.write(init_msg)
        process.stdin.write(launch_msg)
        process.stdin.flush()
        
        # Wait for response or timeout
        try:
            stdout, stderr = process.communicate(timeout=10)
            end_time = time.time()
            
            duration = (end_time - start_time) * 1000
            print(f"Normal LLDB-DAP: {duration:.1f}ms")
            
            if stderr:
                print(f"Stderr: {stderr}")
                
            return duration
            
        except subprocess.TimeoutExpired:
            process.kill()
            print("Normal LLDB-DAP: TIMEOUT (>10s)")
            return 10000
            
    except Exception as e:
        print(f"Normal LLDB-DAP error: {e}")
        return None

def benchmark_lldb_dap_fast(program_path):
    """Benchmark fast launch mode lldb-dap startup."""
    print("=== Benchmarking Fast LLDB-DAP ===")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            ["./build/bin/lldb-dap"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send initialize request
        init_msg = create_dap_message("initialize", {
            "clientID": "benchmark",
            "adapterID": "lldb-dap"
        })
        
        # Send launch request with fast options
        launch_msg = create_dap_message("launch", {
            "program": str(program_path),
            "stopOnEntry": True,
            "fastLaunchMode": True,
            "deferSymbolLoading": True,
            "lazyPluginLoading": True,
            "debuginfodTimeoutMs": 1000,
            "disableNetworkSymbols": True
        })
        
        process.stdin.write(init_msg)
        process.stdin.write(launch_msg)
        process.stdin.flush()
        
        # Wait for response or timeout
        try:
            stdout, stderr = process.communicate(timeout=10)
            end_time = time.time()
            
            duration = (end_time - start_time) * 1000
            print(f"Fast LLDB-DAP: {duration:.1f}ms")
            
            if stderr:
                print(f"Stderr: {stderr}")
                
            return duration
            
        except subprocess.TimeoutExpired:
            process.kill()
            print("Fast LLDB-DAP: TIMEOUT (>10s)")
            return 10000
            
    except Exception as e:
        print(f"Fast LLDB-DAP error: {e}")
        return None

def benchmark_gdb(program_path):
    """Benchmark GDB startup for comparison."""
    print("=== Benchmarking GDB ===")
    
    start_time = time.time()
    
    try:
        # Simple GDB startup test
        process = subprocess.Popen(
            ["gdb", "--batch", "--ex", "run", "--ex", "quit", str(program_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(timeout=10)
        end_time = time.time()
        
        duration = (end_time - start_time) * 1000
        print(f"GDB: {duration:.1f}ms")
        
        return duration
        
    except subprocess.TimeoutExpired:
        process.kill()
        print("GDB: TIMEOUT (>10s)")
        return 10000
    except FileNotFoundError:
        print("GDB: Not found")
        return None
    except Exception as e:
        print(f"GDB error: {e}")
        return None

def main():
    """Run performance benchmarks."""
    print("LLDB-DAP Performance Benchmark")
    print("=" * 50)
    
    # Create test program
    try:
        program_path = create_test_program()
        print(f"Created test program: {program_path}")
    except Exception as e:
        print(f"Failed to create test program: {e}")
        return 1
    
    # Run benchmarks
    results = {}
    
    # Benchmark normal lldb-dap
    results['normal_lldb_dap'] = benchmark_lldb_dap_normal(program_path)
    
    # Benchmark fast lldb-dap
    results['fast_lldb_dap'] = benchmark_lldb_dap_fast(program_path)
    
    # Benchmark GDB for comparison
    results['gdb'] = benchmark_gdb(program_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS:")
    print("=" * 50)
    
    for name, duration in results.items():
        if duration is not None:
            print(f"{name:20}: {duration:6.1f}ms")
        else:
            print(f"{name:20}: FAILED")
    
    # Analysis
    if results['normal_lldb_dap'] and results['gdb']:
        ratio = results['normal_lldb_dap'] / results['gdb']
        print(f"\nNormal LLDB-DAP is {ratio:.1f}x slower than GDB")
    
    if results['fast_lldb_dap'] and results['normal_lldb_dap']:
        improvement = ((results['normal_lldb_dap'] - results['fast_lldb_dap']) / results['normal_lldb_dap']) * 100
        print(f"Fast mode improves performance by {improvement:.1f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
