namespace distribution_explorer
{
    partial class distexAboutBox
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
          this.tableLayoutPanel = new System.Windows.Forms.TableLayoutPanel();
          this.logoPictureBox = new System.Windows.Forms.PictureBox();
          this.labelProductName = new System.Windows.Forms.Label();
          this.labelVersion = new System.Windows.Forms.Label();
          this.labelCopyright = new System.Windows.Forms.Label();
          this.labelCompanyName = new System.Windows.Forms.Label();
          this.textBoxDescription = new System.Windows.Forms.TextBox();
          this.okButton = new System.Windows.Forms.Button();
          this.tableLayoutPanel.SuspendLayout();
          ((System.ComponentModel.ISupportInitialize)(this.logoPictureBox)).BeginInit();
          this.SuspendLayout();
          // 
          // tableLayoutPanel
          // 
          this.tableLayoutPanel.ColumnCount = 2;
          this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 33F));
          this.tableLayoutPanel.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 67F));
          this.tableLayoutPanel.Controls.Add(this.logoPictureBox, 0, 0);
          this.tableLayoutPanel.Controls.Add(this.labelProductName, 1, 0);
          this.tableLayoutPanel.Controls.Add(this.labelVersion, 1, 1);
          this.tableLayoutPanel.Controls.Add(this.labelCopyright, 1, 2);
          this.tableLayoutPanel.Controls.Add(this.labelCompanyName, 1, 3);
          this.tableLayoutPanel.Controls.Add(this.textBoxDescription, 1, 4);
          this.tableLayoutPanel.Controls.Add(this.okButton, 1, 5);
          this.tableLayoutPanel.Dock = System.Windows.Forms.DockStyle.Fill;
          this.tableLayoutPanel.Location = new System.Drawing.Point(12, 11);
          this.tableLayoutPanel.Margin = new System.Windows.Forms.Padding(4);
          this.tableLayoutPanel.Name = "tableLayoutPanel";
          this.tableLayoutPanel.RowCount = 6;
          this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10F));
          this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10F));
          this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10F));
          this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10F));
          this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
          this.tableLayoutPanel.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10F));
          this.tableLayoutPanel.Size = new System.Drawing.Size(556, 326);
          this.tableLayoutPanel.TabIndex = 0;
          // 
          // logoPictureBox
          // 
          this.logoPictureBox.Dock = System.Windows.Forms.DockStyle.Fill;
          this.logoPictureBox.Image = global::distribution_explorer.Properties.Resources.ToolkitLogo;
          this.logoPictureBox.InitialImage = global::distribution_explorer.Properties.Resources.ToolkitLogo;
          this.logoPictureBox.Location = new System.Drawing.Point(4, 4);
          this.logoPictureBox.Margin = new System.Windows.Forms.Padding(4);
          this.logoPictureBox.Name = "logoPictureBox";
          this.tableLayoutPanel.SetRowSpan(this.logoPictureBox, 6);
          this.logoPictureBox.Size = new System.Drawing.Size(175, 318);
          this.logoPictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
          this.logoPictureBox.TabIndex = 12;
          this.logoPictureBox.TabStop = false;
          // 
          // labelProductName
          // 
          this.labelProductName.Dock = System.Windows.Forms.DockStyle.Fill;
          this.labelProductName.Location = new System.Drawing.Point(191, 0);
          this.labelProductName.Margin = new System.Windows.Forms.Padding(8, 0, 4, 0);
          this.labelProductName.MaximumSize = new System.Drawing.Size(0, 21);
          this.labelProductName.Name = "labelProductName";
          this.labelProductName.Size = new System.Drawing.Size(361, 21);
          this.labelProductName.TabIndex = 19;
          this.labelProductName.Text = "Product Name";
          this.labelProductName.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
          // 
          // labelVersion
          // 
          this.labelVersion.Dock = System.Windows.Forms.DockStyle.Fill;
          this.labelVersion.Location = new System.Drawing.Point(191, 32);
          this.labelVersion.Margin = new System.Windows.Forms.Padding(8, 0, 4, 0);
          this.labelVersion.MaximumSize = new System.Drawing.Size(0, 21);
          this.labelVersion.Name = "labelVersion";
          this.labelVersion.Size = new System.Drawing.Size(361, 21);
          this.labelVersion.TabIndex = 0;
          this.labelVersion.Text = "Version";
          this.labelVersion.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
          // 
          // labelCopyright
          // 
          this.labelCopyright.Dock = System.Windows.Forms.DockStyle.Fill;
          this.labelCopyright.Location = new System.Drawing.Point(191, 64);
          this.labelCopyright.Margin = new System.Windows.Forms.Padding(8, 0, 4, 0);
          this.labelCopyright.MaximumSize = new System.Drawing.Size(0, 21);
          this.labelCopyright.Name = "labelCopyright";
          this.labelCopyright.Size = new System.Drawing.Size(361, 21);
          this.labelCopyright.TabIndex = 21;
          this.labelCopyright.Text = "Copyright";
          this.labelCopyright.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
          // 
          // labelCompanyName
          // 
          this.labelCompanyName.Dock = System.Windows.Forms.DockStyle.Fill;
          this.labelCompanyName.Location = new System.Drawing.Point(191, 96);
          this.labelCompanyName.Margin = new System.Windows.Forms.Padding(8, 0, 4, 0);
          this.labelCompanyName.MaximumSize = new System.Drawing.Size(0, 21);
          this.labelCompanyName.Name = "labelCompanyName";
          this.labelCompanyName.Size = new System.Drawing.Size(361, 21);
          this.labelCompanyName.TabIndex = 22;
          this.labelCompanyName.Text = "Company Name";
          this.labelCompanyName.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
          // 
          // textBoxDescription
          // 
          this.textBoxDescription.Dock = System.Windows.Forms.DockStyle.Fill;
          this.textBoxDescription.Location = new System.Drawing.Point(191, 132);
          this.textBoxDescription.Margin = new System.Windows.Forms.Padding(8, 4, 4, 4);
          this.textBoxDescription.Multiline = true;
          this.textBoxDescription.Name = "textBoxDescription";
          this.textBoxDescription.ReadOnly = true;
          this.textBoxDescription.ScrollBars = System.Windows.Forms.ScrollBars.Both;
          this.textBoxDescription.Size = new System.Drawing.Size(361, 155);
          this.textBoxDescription.TabIndex = 23;
          this.textBoxDescription.TabStop = false;
          this.textBoxDescription.Text = "Description";
          // 
          // okButton
          // 
          this.okButton.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
          this.okButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
          this.okButton.Location = new System.Drawing.Point(452, 295);
          this.okButton.Margin = new System.Windows.Forms.Padding(4);
          this.okButton.Name = "okButton";
          this.okButton.Size = new System.Drawing.Size(100, 27);
          this.okButton.TabIndex = 24;
          this.okButton.Text = "&OK";
          // 
          // distexAboutBox
          // 
          this.AcceptButton = this.okButton;
          this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
          this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
          this.ClientSize = new System.Drawing.Size(580, 348);
          this.Controls.Add(this.tableLayoutPanel);
          this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
          this.Margin = new System.Windows.Forms.Padding(4);
          this.MaximizeBox = false;
          this.MinimizeBox = false;
          this.Name = "distexAboutBox";
          this.Padding = new System.Windows.Forms.Padding(12, 11, 12, 11);
          this.ShowIcon = false;
          this.ShowInTaskbar = false;
          this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
          this.Text = "About Statistical Distribution Explorer";
          this.tableLayoutPanel.ResumeLayout(false);
          this.tableLayoutPanel.PerformLayout();
          ((System.ComponentModel.ISupportInitialize)(this.logoPictureBox)).EndInit();
          this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel;
        private System.Windows.Forms.PictureBox logoPictureBox;
        private System.Windows.Forms.Label labelProductName;
        private System.Windows.Forms.Label labelVersion;
        private System.Windows.Forms.Label labelCopyright;
        private System.Windows.Forms.Label labelCompanyName;
        private System.Windows.Forms.TextBox textBoxDescription;
        private System.Windows.Forms.Button okButton;
    }
}
