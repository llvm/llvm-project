import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class LitTestProvider implements vscode.Disposable {
    private controller: vscode.TestController;
    private testItemRoot: vscode.TestItem;

    constructor(context: vscode.ExtensionContext) {
        // Create the TestController and root test item
        this.controller = vscode.tests.createTestController('litTestController', 'LIT Tests');
        this.testItemRoot = this.controller.createTestItem('litTestsRoot', 'LIT Tests');
        this.controller.items.add(this.testItemRoot);
        context.subscriptions.push(this.controller);

        // Discover tests initially on extension activation
        this.discoverLitTests();

        // Set up file open listener for MLIR files
        vscode.workspace.onDidOpenTextDocument(document => {
            if (document.uri.fsPath.endsWith('.mlir')) {
                console.log(`MLIR file opened: ${document.uri.fsPath}`);
                this.discoverLitTests();
            }
        });

        // Set up run profile for running tests
        this.controller.createRunProfile('Run Tests', vscode.TestRunProfileKind.Run, async (request, token) => {
            const run = this.controller.createTestRun(request);
            for (const test of request.include ?? []) {
                await this.runTest(run, test, token);
            }
            run.end();
        });
    }

    // Function to prompt the user to re-enter the LIT tool path and test folder path
    public async reconfigureLitSettings() {
        const config = vscode.workspace.getConfiguration('lit');

        // Prompt for the lit tool path and update the configuration
        const litToolPath = await this.promptForPath('Please re-enter the path to the lit tool');
        if (litToolPath) {
            await config.update('lit_path', litToolPath, vscode.ConfigurationTarget.Workspace);
        }

        // Prompt for the test folder path and update the configuration
        const testFolderPath = await this.promptForPath('Please re-enter the path to the folder containing LIT tests');
        if (testFolderPath) {
            await config.update('test_root_folder', testFolderPath, vscode.ConfigurationTarget.Workspace);
        }

        // Rediscover tests after reconfiguration
        this.discoverLitTests();
    }

    // Function to discover LIT tests and display them in the test explorer
    private async discoverLitTests() {
        const config = vscode.workspace.getConfiguration('lit');

        // Get the lit tool path and test folder from the config
        let litToolPath = config.get<string>('lit_path');
        let testFolderPath = config.get<string>('test_root_folder');

        // If the lit tool path or test folder is not set, prompt the user to enter them
        if (!litToolPath) {
            litToolPath = await this.promptForPath('Please enter the path to the lit tool');
            if (litToolPath) {
                await config.update('lit_path', litToolPath, vscode.ConfigurationTarget.Workspace);
            }
        }

        if (!testFolderPath) {
            testFolderPath = await this.promptForPath('Please enter the path to the folder containing LIT tests');
            if (testFolderPath) {
                await config.update('test_root_folder', testFolderPath, vscode.ConfigurationTarget.Workspace);
            }
        }

        // Ensure both values are now set before proceeding
        if (!litToolPath || !testFolderPath) {
            vscode.window.showErrorMessage('LIT tool path or test folder path not set. Test discovery cannot proceed.');
            return;
        }

        // Ensure the test folder path is absolute (relative to workspace)
        const absoluteTestFolderPath = path.isAbsolute(testFolderPath)
            ? testFolderPath
            : path.join(vscode.workspace.workspaceFolders?.[0].uri.fsPath || '', testFolderPath);

        if (!fs.existsSync(absoluteTestFolderPath)) {
            vscode.window.showErrorMessage(`Test folder not found: ${absoluteTestFolderPath}`);
            return;
        }

        // Clear previous test items before discovering new ones
        this.testItemRoot.children.replace([]); // Use replace([]) to clear the test items

        // Recursively scan the folder for LIT tests
        this.scanDirectory(this.testItemRoot, absoluteTestFolderPath);
    }

    // Function to scan a directory for LIT tests
    private scanDirectory(parent: vscode.TestItem, directory: string) {
        const items = fs.readdirSync(directory, { withFileTypes: true });

        for (const item of items) {
            const itemPath = path.join(directory, item.name);

            if (item.isDirectory()) {
                // Create a new TestItem for the directory
                const dirTestItem = this.controller.createTestItem(itemPath, item.name);
                parent.children.add(dirTestItem);

                // Recursively scan this subdirectory
                this.scanDirectory(dirTestItem, itemPath);
            } else if (item.isFile() && this.isLitTestFile(item.name)) {
                // It's a file and we assume it's a LIT test file
                const testItem = this.controller.createTestItem(itemPath, item.name, vscode.Uri.file(itemPath));
                parent.children.add(testItem);
            }
        }
    }

    // A simple helper function to check if a file is a LIT test (now checks for .mlir files)
    private isLitTestFile(filename: string): boolean {
        return filename.endsWith('.mlir'); // Now only checks for .mlir files
    }

    // Function to run a LIT test
    private async runTest(run: vscode.TestRun, test: vscode.TestItem, token: vscode.CancellationToken) {
        run.started(test);

        const config = vscode.workspace.getConfiguration('lit');
        const litToolPath = config.get<string>('lit_path') || 'lit';  // Default to 'lit'

        try {
            const result = await this.runLitTest(litToolPath, test.uri!.fsPath);
            if (result.passed) {
                run.passed(test);
            } else {
                run.failed(test, new vscode.TestMessage(result.errorMessage));
            }
        } catch (error) {
            run.errored(test, new vscode.TestMessage(error.message));
        }

        run.end();
    }

    // Function to execute the LIT test using the lit tool
    private async runLitTest(litToolPath: string, testPath: string): Promise<{ passed: boolean, errorMessage?: string }> {
        const { exec } = require('child_process');

        return new Promise((resolve, reject) => {
            exec(`${litToolPath} -v ${testPath}`, (error: any, stdout: string, stderr: string) => {
                if (error) {
                    resolve({ passed: false, errorMessage: `stdout: ${stdout}\stderr: ${stderr}` });
                } else {
                    resolve({ passed: true });
                }
            });
        });
    }

    // Helper function to prompt the user for a path
    private async promptForPath(promptMessage: string): Promise<string | undefined> {
        return vscode.window.showInputBox({
            prompt: promptMessage,
            placeHolder: 'Enter a valid path'
        });
    }

    // Implementing the dispose method to clean up resources
    public dispose() {
        this.controller.dispose();  // Dispose of the TestController
    }
}
